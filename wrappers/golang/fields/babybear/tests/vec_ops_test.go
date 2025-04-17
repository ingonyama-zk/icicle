package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	babybear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/babybear/vecOps"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testBabybearVecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := babybear.GenerateScalars(testSize)
	b := babybear.GenerateScalars(testSize)
	var scalar babybear.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[babybear.ScalarField], testSize)
	out2 := make(core.HostSlice[babybear.ScalarField], testSize)
	out3 := make(core.HostSlice[babybear.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testBabybearTranspose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := babybear.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[babybear.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[babybear.ScalarField], rowSize*columnSize)

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
	output := make(core.HostSlice[babybear.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testBabybearSum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := babybear.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[babybear.ScalarField], batchSize)
	result2 := make(core.HostSlice[babybear.ScalarField], batchSize)

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

func testBabybearProduct(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := babybear.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[babybear.ScalarField], batchSize)
	result2 := make(core.HostSlice[babybear.ScalarField], batchSize)

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

func testBabybearInverse(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := babybear.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[babybear.ScalarField], batchSize)
	result2 := make(core.HostSlice[babybear.ScalarField], batchSize)

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

type BabybearVecOpsTestSuite struct {
	suite.Suite
}

func (s *BabybearVecOpsTestSuite) TestBabybearVecOps() {
	s.Run("TestBabybearVecOps", testWrapper(&s.Suite, testBabybearVecOps))
	s.Run("TestBabybearTranspose", testWrapper(&s.Suite, testBabybearTranspose))
	s.Run("TestBabybearSum", testWrapper(&s.Suite, testBabybearSum))
	s.Run("TestBabybearProduct", testWrapper(&s.Suite, testBabybearProduct))
}

func TestSuiteBabybearVecOps(t *testing.T) {
	suite.Run(t, new(BabybearVecOpsTestSuite))
}
