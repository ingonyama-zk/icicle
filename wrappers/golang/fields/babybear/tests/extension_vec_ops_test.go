package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	babybear_extension "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/extension"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/extension/vecOps"
	"github.com/stretchr/testify/assert"
)

func TestBabybear_extensionVecOps(t *testing.T) {
	testSize := 1 << 14

	a := babybear_extension.GenerateScalars(testSize)
	b := babybear_extension.GenerateScalars(testSize)
	var scalar babybear_extension.ExtensionField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[babybear_extension.ExtensionField], testSize)
	out2 := make(core.HostSlice[babybear_extension.ExtensionField], testSize)
	out3 := make(core.HostSlice[babybear_extension.ExtensionField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	assert.Equal(t, a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	assert.Equal(t, a, out3)
}

func TestBabybear_extensionTranspose(t *testing.T) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := babybear_extension.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[babybear_extension.ExtensionField], rowSize*columnSize)
	out2 := make(core.HostSlice[babybear_extension.ExtensionField], rowSize*columnSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.TransposeMatrix(matrix, out, columnSize, rowSize, cfg)
	vecOps.TransposeMatrix(out, out2, rowSize, columnSize, cfg)

	assert.Equal(t, matrix, out2)

	var dMatrix, dOut, dOut2 core.DeviceSlice

	matrix.CopyToDevice(&dMatrix, true)
	dOut.Malloc(matrix.SizeOfElement(), columnSize*rowSize)
	dOut2.Malloc(matrix.SizeOfElement(), columnSize*rowSize)

	vecOps.TransposeMatrix(dMatrix, dOut, columnSize, rowSize, cfg)
	vecOps.TransposeMatrix(dOut, dOut2, rowSize, columnSize, cfg)
	output := make(core.HostSlice[babybear_extension.ExtensionField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	assert.Equal(t, matrix, output)
}
