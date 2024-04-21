package vecOps

import (
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	babybear_extension "github.com/ingonyama-zk/icicle/wrappers/golang/fields/babybear/extension"
	"github.com/stretchr/testify/assert"
)

func TestVecOps(t *testing.T) {
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

	matrix := babybear_extension.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[babybear_extension.ExtensionField], rowSize*columnSize)
	out2 := make(core.HostSlice[babybear_extension.ExtensionField], rowSize*columnSize)

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
	output := make(core.HostSlice[babybear_extension.ExtensionField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	assert.Equal(t, matrix, output)
}
