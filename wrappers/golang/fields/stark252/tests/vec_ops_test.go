package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	stark252 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/stark252"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/stark252/vecOps"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testStark252VecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := stark252.GenerateScalars(testSize)
	b := stark252.GenerateScalars(testSize)
	var scalar stark252.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[stark252.ScalarField], testSize)
	out2 := make(core.HostSlice[stark252.ScalarField], testSize)
	out3 := make(core.HostSlice[stark252.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testStark252Transpose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := stark252.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[stark252.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[stark252.ScalarField], rowSize*columnSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.TransposeMatrix(matrix, out, columnSize, rowSize, cfg)
	vecOps.TransposeMatrix(out, out2, rowSize, columnSize, cfg)

	suite.Equal(matrix, out2)

	var dMatrix, dOut, dOut2 core.DeviceSlice

	matrix.CopyToDevice(&dMatrix, true)
	dOut.Malloc(matrix.SizeOfElement(), columnSize*rowSize)
	dOut2.Malloc(matrix.SizeOfElement(), columnSize*rowSize)

	vecOps.TransposeMatrix(dMatrix, dOut, columnSize, rowSize, cfg)
	vecOps.TransposeMatrix(dOut, dOut2, rowSize, columnSize, cfg)
	output := make(core.HostSlice[stark252.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testStark252Sum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := stark252.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[stark252.ScalarField], batchSize)
	result2 := make(core.HostSlice[stark252.ScalarField], batchSize)

	cfg := core.DefaultVecOpsConfig()
	cfg.BatchSize = int32(batchSize)

	// CPU run
	test_helpers.ActivateReferenceDevice()
	vecOps.ReductionVecOp(a, result, cfg, core.Sum)

	// Cuda run
	test_helpers.ActivateMainDevice()
	var dA, dResult core.DeviceSlice
	a.CopyToDevice(&dA, true)
	dResult.Malloc(a.SizeOfElement()*batchSize, batchSize)

	vecOps.ReductionVecOp(dA, dResult, cfg, core.Sum)
	result2.CopyFromDevice(&dResult)

	suite.Equal(result, result2)
}

func testStark252Product(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := stark252.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[stark252.ScalarField], batchSize)
	result2 := make(core.HostSlice[stark252.ScalarField], batchSize)

	cfg := core.DefaultVecOpsConfig()
	cfg.BatchSize = int32(batchSize)

	// CPU run
	test_helpers.ActivateReferenceDevice()
	vecOps.ReductionVecOp(a, result, cfg, core.Product)

	// Cuda run
	test_helpers.ActivateMainDevice()
	var dA, dResult core.DeviceSlice
	a.CopyToDevice(&dA, true)
	dResult.Malloc(a.SizeOfElement()*batchSize, batchSize)

	vecOps.ReductionVecOp(dA, dResult, cfg, core.Product)
	result2.CopyFromDevice(&dResult)

	suite.Equal(result, result2)
}

type Stark252VecOpsTestSuite struct {
	suite.Suite
}

func (s *Stark252VecOpsTestSuite) TestStark252VecOps() {
	s.Run("TestStark252VecOps", test_helpers.TestWrapper(&s.Suite, testStark252VecOps))
	s.Run("TestStark252Transpose", test_helpers.TestWrapper(&s.Suite, testStark252Transpose))
	s.Run("TestStark252Sum", test_helpers.TestWrapper(&s.Suite, testStark252Sum))
	s.Run("TestStark252Product", test_helpers.TestWrapper(&s.Suite, testStark252Product))
}

func TestSuiteStark252VecOps(t *testing.T) {
	suite.Run(t, new(Stark252VecOpsTestSuite))
}
