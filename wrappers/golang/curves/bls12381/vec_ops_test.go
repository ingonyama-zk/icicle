package bls12381

import (
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	"github.com/stretchr/testify/assert"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func TestVecOps(t *testing.T) {
	testSize := 1 << 14

	a := GenerateScalars(testSize)
	b := GenerateScalars(testSize)
	var scalar ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[ScalarField], testSize)
	out2 := make(core.HostSlice[ScalarField], testSize)
	out3 := make(core.HostSlice[ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	VecOp[ScalarField](a, b, out, cfg, core.Add)
	VecOp[ScalarField](out, b, out2, cfg, core.Sub)

	assert.Equal(t, a, out2)

	VecOp[ScalarField](a, ones, out3, cfg, core.Mul)

	assert.Equal(t, ones, out3)
}


func TestTranspose(t *testing.T) {

	rowSize := 1 << 6
	columnSize := 1 << 8
	onDevice := false
	isAsync := true

	matrix := GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[ScalarField], rowSize*columnSize)

	ctx, _ := cr.GetDefaultDeviceContext()

	TransposeMatrix(matrix, out, columnSize, rowSize, ctx, onDevice, isAsync)
	TransposeMatrix(out, out2, rowSize, columnSize, ctx, onDevice, isAsync)
	
	assert.Equal(t, matrix, out2)

	var d_matrix, d_out, d_out2 core.DeviceSlice
	onDevice = true

	matrix.CopyToDeviceAsync(&d_matrix, *ctx.Stream, true)
	d_out.MallocAsync(columnSize*rowSize*matrix.SizeOfElement(), matrix.SizeOfElement(), *ctx.Stream)
	d_out2.MallocAsync(columnSize*rowSize*matrix.SizeOfElement(), matrix.SizeOfElement(), *ctx.Stream)

	TransposeMatrix(d_matrix, d_out, columnSize, rowSize, ctx, onDevice, isAsync)
	TransposeMatrix(d_out, d_out2, rowSize, columnSize, ctx, onDevice, isAsync)
	output := make(core.HostSlice[ScalarField], rowSize*columnSize)
	output.CopyFromDeviceAsync(&d_out2, *ctx.Stream)

	assert.Equal(t, matrix, output)
}