package bw6761

import (
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	"github.com/stretchr/testify/assert"
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

	matrix := GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[ScalarField], rowSize*columnSize)

	ctx, _ := cr.GetDefaultDeviceContext()
	stream, _ := cr.CreateStream()
	ctx.Stream = &stream

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
