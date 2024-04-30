package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	babybear_extension "github.com/ingonyama-zk/icicle/v2/wrappers/golang/fields/babybear/extension"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/fields/babybear/extension/vecOps"
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

	cfg := core.DefaultVecOpsConfigForDevice(0)

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	assert.Equal(t, a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	assert.Equal(t, a, out3)
}

func TestBabybear_extensionTranspose(t *testing.T) {
	rowSize := 1 << 6
	columnSize := 1 << 8
	onDevice := false
	isAsync := false

	matrix := babybear_extension.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[babybear_extension.ExtensionField], rowSize*columnSize)
	out2 := make(core.HostSlice[babybear_extension.ExtensionField], rowSize*columnSize)

	ctx, _ := cr.GetDefaultContextForDevice(0)

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
	output := make(core.HostSlice[babybear_extension.ExtensionField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	assert.Equal(t, matrix, output)
}
