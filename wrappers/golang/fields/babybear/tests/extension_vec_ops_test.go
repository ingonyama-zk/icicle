package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	babybearExtension "github.com/ingonyama-zk/icicle/wrappers/golang/fields/babybear/extension"
	"github.com/ingonyama-zk/icicle/wrappers/golang/fields/babybear/extension/vecOps"
	"github.com/stretchr/testify/assert"
)

func TestBabybearExtensionVecOps(t *testing.T) {
	testSize := 1 << 14

	a := babybearExtension.GenerateScalars(testSize)
	b := babybearExtension.GenerateScalars(testSize)
	var scalar babybearExtension.ExtensionField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[babybearExtension.ExtensionField], testSize)
	out2 := make(core.HostSlice[babybearExtension.ExtensionField], testSize)
	out3 := make(core.HostSlice[babybearExtension.ExtensionField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	assert.Equal(t, a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	assert.Equal(t, a, out3)
}

func TestBabybearExtensionTranspose(t *testing.T) {
	rowSize := 1 << 6
	columnSize := 1 << 8
	onDevice := false
	isAsync := false

	matrix := babybearExtension.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[babybearExtension.ExtensionField], rowSize*columnSize)
	out2 := make(core.HostSlice[babybearExtension.ExtensionField], rowSize*columnSize)

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
	output := make(core.HostSlice[babybearExtension.ExtensionField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	assert.Equal(t, matrix, output)
}
