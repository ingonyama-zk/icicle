package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	koalabear "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/koalabear"
	koalabear_extension "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/koalabear/extension"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/koalabear/extension/vecOps"
	"github.com/stretchr/testify/suite"
)

func testKoalabear_extensionVecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := koalabear_extension.GenerateScalars(testSize)
	b := koalabear_extension.GenerateScalars(testSize)
	var scalar koalabear_extension.ExtensionField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[koalabear_extension.ExtensionField], testSize)
	out2 := make(core.HostSlice[koalabear_extension.ExtensionField], testSize)
	out3 := make(core.HostSlice[koalabear_extension.ExtensionField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testKoalabear_extensionTranspose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := koalabear_extension.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[koalabear_extension.ExtensionField], rowSize*columnSize)
	out2 := make(core.HostSlice[koalabear_extension.ExtensionField], rowSize*columnSize)

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
	output := make(core.HostSlice[koalabear_extension.ExtensionField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testKoalabear_extensionSum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := koalabear_extension.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[koalabear_extension.ExtensionField], batchSize)
	result2 := make(core.HostSlice[koalabear_extension.ExtensionField], batchSize)

	cfg := core.DefaultVecOpsConfig()
	cfg.BatchSize = int32(batchSize)

	vecOps.SumScalars(a, result, cfg)

	// Test with device memory
	var dA, dResult core.DeviceSlice
	a.CopyToDevice(&dA, true)
	dResult.Malloc(a.SizeOfElement()*batchSize, batchSize)

	vecOps.SumScalars(dA, dResult, cfg)
	result2.CopyFromDevice(&dResult)

	suite.Equal(result, result2)
}

func testKoalabear_extensionProduct(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := koalabear_extension.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[koalabear_extension.ExtensionField], batchSize)
	result2 := make(core.HostSlice[koalabear_extension.ExtensionField], batchSize)

	cfg := core.DefaultVecOpsConfig()
	cfg.BatchSize = int32(batchSize)

	vecOps.ProductScalars(a, result, cfg)

	// Test with device memory
	var dA, dResult core.DeviceSlice
	a.CopyToDevice(&dA, true)
	dResult.Malloc(a.SizeOfElement()*batchSize, batchSize)

	vecOps.ProductScalars(dA, dResult, cfg)
	result2.CopyFromDevice(&dResult)

	suite.Equal(result, result2)
}

func testKoalabear_extensionMixedVecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := koalabear_extension.GenerateScalars(testSize)
	var scalar koalabear.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[koalabear_extension.ExtensionField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.MixedVecOp(a, ones, out, cfg, core.Mul)

	suite.Equal(a, out)
}

type Koalabear_extensionVecOpsTestSuite struct {
	suite.Suite
}

func (s *Koalabear_extensionVecOpsTestSuite) TestKoalabear_extensionVecOps() {
	s.Run("TestKoalabear_extensionVecOps", testWrapper(&s.Suite, testKoalabear_extensionVecOps))
	s.Run("TestKoalabear_extensionTranspose", testWrapper(&s.Suite, testKoalabear_extensionTranspose))
	s.Run("TestKoalabear_extensionSum", testWrapper(&s.Suite, testKoalabear_extensionSum))
	s.Run("TestKoalabear_extensionProduct", testWrapper(&s.Suite, testKoalabear_extensionProduct))
	s.Run("TestKoalabear_extensionMixedVecOps", testWrapper(&s.Suite, testKoalabear_extensionMixedVecOps))
}

func TestSuiteKoalabear_extensionVecOps(t *testing.T) {
	suite.Run(t, new(Koalabear_extensionVecOpsTestSuite))
}
