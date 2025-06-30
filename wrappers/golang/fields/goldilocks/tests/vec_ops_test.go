package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	goldilocks "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/goldilocks"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/goldilocks/vecOps"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testGoldilocksVecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := goldilocks.GenerateScalars(testSize)
	b := goldilocks.GenerateScalars(testSize)
	var scalar goldilocks.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[goldilocks.ScalarField], testSize)
	out2 := make(core.HostSlice[goldilocks.ScalarField], testSize)
	out3 := make(core.HostSlice[goldilocks.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testGoldilocksTranspose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := goldilocks.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[goldilocks.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[goldilocks.ScalarField], rowSize*columnSize)

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
	output := make(core.HostSlice[goldilocks.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testGoldilocksSum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := goldilocks.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[goldilocks.ScalarField], batchSize)
	result2 := make(core.HostSlice[goldilocks.ScalarField], batchSize)

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

func testGoldilocksProduct(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := goldilocks.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[goldilocks.ScalarField], batchSize)
	result2 := make(core.HostSlice[goldilocks.ScalarField], batchSize)

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

type GoldilocksVecOpsTestSuite struct {
	suite.Suite
}

func (s *GoldilocksVecOpsTestSuite) TestGoldilocksVecOps() {
	s.Run("TestGoldilocksVecOps", test_helpers.TestWrapper(&s.Suite, testGoldilocksVecOps))
	s.Run("TestGoldilocksTranspose", test_helpers.TestWrapper(&s.Suite, testGoldilocksTranspose))
	s.Run("TestGoldilocksSum", test_helpers.TestWrapper(&s.Suite, testGoldilocksSum))
	s.Run("TestGoldilocksProduct", test_helpers.TestWrapper(&s.Suite, testGoldilocksProduct))
}

func TestSuiteGoldilocksVecOps(t *testing.T) {
	suite.Run(t, new(GoldilocksVecOpsTestSuite))
}
