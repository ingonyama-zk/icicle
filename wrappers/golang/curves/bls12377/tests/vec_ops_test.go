package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	bls12_377 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bls12377/vecOps"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testBls12_377VecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := bls12_377.GenerateScalars(testSize)
	b := bls12_377.GenerateScalars(testSize)
	var scalar bls12_377.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bls12_377.ScalarField], testSize)
	out2 := make(core.HostSlice[bls12_377.ScalarField], testSize)
	out3 := make(core.HostSlice[bls12_377.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testBls12_377Transpose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := bls12_377.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[bls12_377.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[bls12_377.ScalarField], rowSize*columnSize)

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
	output := make(core.HostSlice[bls12_377.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testBls12_377Sum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := bls12_377.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[bls12_377.ScalarField], batchSize)
	result2 := make(core.HostSlice[bls12_377.ScalarField], batchSize)

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

func testBls12_377Product(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := bls12_377.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[bls12_377.ScalarField], batchSize)
	result2 := make(core.HostSlice[bls12_377.ScalarField], batchSize)

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

type Bls12_377VecOpsTestSuite struct {
	suite.Suite
}

func (s *Bls12_377VecOpsTestSuite) TestBls12_377VecOps() {
	s.Run("TestBls12_377VecOps", test_helpers.TestWrapper(&s.Suite, testBls12_377VecOps))
	s.Run("TestBls12_377Transpose", test_helpers.TestWrapper(&s.Suite, testBls12_377Transpose))
	s.Run("TestBls12_377Sum", test_helpers.TestWrapper(&s.Suite, testBls12_377Sum))
	s.Run("TestBls12_377Product", test_helpers.TestWrapper(&s.Suite, testBls12_377Product))
}

func TestSuiteBls12_377VecOps(t *testing.T) {
	suite.Run(t, new(Bls12_377VecOpsTestSuite))
}
