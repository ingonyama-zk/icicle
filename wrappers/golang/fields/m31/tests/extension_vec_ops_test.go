package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	m31 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/m31"
	m31_extension "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/m31/extension"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/m31/extension/vecOps"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	"github.com/stretchr/testify/suite"
)

func testM31_extensionVecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := m31_extension.GenerateScalars(testSize)
	b := m31_extension.GenerateScalars(testSize)
	var scalar m31_extension.ExtensionField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[m31_extension.ExtensionField], testSize)
	out2 := make(core.HostSlice[m31_extension.ExtensionField], testSize)
	out3 := make(core.HostSlice[m31_extension.ExtensionField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	suite.Equal(a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	suite.Equal(a, out3)
}

func testM31_extensionTranspose(suite *suite.Suite) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := m31_extension.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[m31_extension.ExtensionField], rowSize*columnSize)
	out2 := make(core.HostSlice[m31_extension.ExtensionField], rowSize*columnSize)

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
	output := make(core.HostSlice[m31_extension.ExtensionField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	suite.Equal(matrix, output)
}

func testM31_extensionSum(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := m31_extension.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[m31_extension.ExtensionField], batchSize)
	result2 := make(core.HostSlice[m31_extension.ExtensionField], batchSize)

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

func testM31_extensionProduct(suite *suite.Suite) {
	testSize := 1 << 14
	batchSize := 3

	a := m31_extension.GenerateScalars(testSize * batchSize)
	result := make(core.HostSlice[m31_extension.ExtensionField], batchSize)
	result2 := make(core.HostSlice[m31_extension.ExtensionField], batchSize)

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

func testM31_extensionMixedVecOps(suite *suite.Suite) {
	testSize := 1 << 14

	a := m31_extension.GenerateScalars(testSize)
	var scalar m31.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[m31_extension.ExtensionField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.MixedVecOp(a, ones, out, cfg, core.Mul)

	suite.Equal(a, out)
}

type M31_extensionVecOpsTestSuite struct {
	suite.Suite
}

func (s *M31_extensionVecOpsTestSuite) TestM31_extensionVecOps() {
	s.Run("TestM31_extensionVecOps", test_helpers.TestWrapper(&s.Suite, testM31_extensionVecOps))
	s.Run("TestM31_extensionTranspose", test_helpers.TestWrapper(&s.Suite, testM31_extensionTranspose))
	s.Run("TestM31_extensionSum", test_helpers.TestWrapper(&s.Suite, testM31_extensionSum))
	s.Run("TestM31_extensionProduct", test_helpers.TestWrapper(&s.Suite, testM31_extensionProduct))
	s.Run("TestM31_extensionMixedVecOps", test_helpers.TestWrapper(&s.Suite, testM31_extensionMixedVecOps))
}

func TestSuiteM31_extensionVecOps(t *testing.T) {
	suite.Run(t, new(M31_extensionVecOpsTestSuite))
}
