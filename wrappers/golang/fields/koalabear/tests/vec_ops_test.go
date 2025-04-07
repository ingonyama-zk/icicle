package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	koalabear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/koalabear"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/koalabear/vecOps"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testKoalabearVecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := koalabear.GenerateScalars(testSize)
	b := koalabear.GenerateScalars(testSize)
	var scalar koalabear.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[koalabear.ScalarField], testSize)
	out2 := make(core.HostSlice[koalabear.ScalarField], testSize)
	out3 := make(core.HostSlice[koalabear.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testKoalabearTranspose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := koalabear.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[koalabear.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[koalabear.ScalarField], rowSize*columnSize)

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
	output := make(core.HostSlice[koalabear.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testKoalabearSum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := koalabear.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[koalabear.ScalarField], batchSize)
	result2 := make(core.HostSlice[koalabear.ScalarField], batchSize)

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

func testKoalabearProduct(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := koalabear.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[koalabear.ScalarField], batchSize)
	result2 := make(core.HostSlice[koalabear.ScalarField], batchSize)

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

type KoalabearVecOpsTestSuite struct {
	suite.Suite
}

func (s *KoalabearVecOpsTestSuite) TestKoalabearVecOps() {
	s.Run("TestKoalabearVecOps", testWrapper(&s.Suite, testKoalabearVecOps))
	s.Run("TestKoalabearTranspose", testWrapper(&s.Suite, testKoalabearTranspose))
	s.Run("TestKoalabearSum", testWrapper(&s.Suite, testKoalabearSum))
	s.Run("TestKoalabearProduct", testWrapper(&s.Suite, testKoalabearProduct))

}

func TestSuiteKoalabearVecOps(t *testing.T) {
	suite.Run(t, new(KoalabearVecOpsTestSuite))
}
