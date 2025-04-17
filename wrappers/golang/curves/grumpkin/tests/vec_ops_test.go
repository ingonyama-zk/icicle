package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	grumpkin "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin/vecOps"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testGrumpkinVecOps(suite *suite.Suite) {
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

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testGrumpkinTranspose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := grumpkin.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[grumpkin.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[grumpkin.ScalarField], rowSize*columnSize)

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
	output := make(core.HostSlice[grumpkin.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testGrumpkinSum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := grumpkin.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[grumpkin.ScalarField], batchSize)
	result2 := make(core.HostSlice[grumpkin.ScalarField], batchSize)

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

func testGrumpkinProduct(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := grumpkin.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[grumpkin.ScalarField], batchSize)
	result2 := make(core.HostSlice[grumpkin.ScalarField], batchSize)

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

func testGrumpkinInverse(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := grumpkin.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[grumpkin.ScalarField], batchSize)
	result2 := make(core.HostSlice[grumpkin.ScalarField], batchSize)

	cfg := core.DefaultVecOpsConfig()
	cfg.BatchSize = int32(batchSize)

	// CPU run
	test_helpers.ActivateReferenceDevice()
	vecOps.ReductionVecOp(a, result, cfg, core.Inverse)

	// Cuda run
	test_helpers.ActivateMainDevice()
	var dA, dResult core.DeviceSlice
	a.CopyToDevice(&dA, true)
	dResult.Malloc(a.SizeOfElement()*batchSize, batchSize)

	vecOps.ReductionVecOp(dA, dResult, cfg, core.Inverse)
	result2.CopyFromDevice(&dResult)

	suite.Equal(result, result2)
}

type GrumpkinVecOpsTestSuite struct {
	suite.Suite
}

func (s *GrumpkinVecOpsTestSuite) TestGrumpkinVecOps() {
	s.Run("TestGrumpkinVecOps", testWrapper(&s.Suite, testGrumpkinVecOps))
	s.Run("TestGrumpkinTranspose", testWrapper(&s.Suite, testGrumpkinTranspose))
	s.Run("TestGrumpkinSum", testWrapper(&s.Suite, testGrumpkinSum))
	s.Run("TestGrumpkinProduct", testWrapper(&s.Suite, testGrumpkinProduct))
}

func TestSuiteGrumpkinVecOps(t *testing.T) {
	suite.Run(t, new(GrumpkinVecOpsTestSuite))
}
