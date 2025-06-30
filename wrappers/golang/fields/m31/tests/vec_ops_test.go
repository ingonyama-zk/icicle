package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	m31 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/m31"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/m31/vecOps"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testM31VecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := m31.GenerateScalars(testSize)
	b := m31.GenerateScalars(testSize)
	var scalar m31.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[m31.ScalarField], testSize)
	out2 := make(core.HostSlice[m31.ScalarField], testSize)
	out3 := make(core.HostSlice[m31.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testM31Transpose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := m31.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[m31.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[m31.ScalarField], rowSize*columnSize)

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
	output := make(core.HostSlice[m31.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testM31Sum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := m31.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[m31.ScalarField], batchSize)
	result2 := make(core.HostSlice[m31.ScalarField], batchSize)

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

func testM31Product(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := m31.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[m31.ScalarField], batchSize)
	result2 := make(core.HostSlice[m31.ScalarField], batchSize)

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

type M31VecOpsTestSuite struct {
	suite.Suite
}

func (s *M31VecOpsTestSuite) TestM31VecOps() {
	s.Run("TestM31VecOps", test_helpers.TestWrapper(&s.Suite, testM31VecOps))
	s.Run("TestM31Transpose", test_helpers.TestWrapper(&s.Suite, testM31Transpose))
	s.Run("TestM31Sum", test_helpers.TestWrapper(&s.Suite, testM31Sum))
	s.Run("TestM31Product", test_helpers.TestWrapper(&s.Suite, testM31Product))
}

func TestSuiteM31VecOps(t *testing.T) {
	suite.Run(t, new(M31VecOpsTestSuite))
}
