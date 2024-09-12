package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	grumpkin "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin/vecOps"
	"github.com/stretchr/testify/suite"
)

func testGrumpkinVecOps(suite suite.Suite) {
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

func testGrumpkinTranspose(suite suite.Suite) {
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

type VecOpsTestSuite struct {
	suite.Suite
}

func (s *VecOpsTestSuite) TestGrumpkinVecOps() {
	s.Run("TestGrumpkinVecOps", testWrapper(s.Suite, testGrumpkinVecOps))
	s.Run("TestGrumpkinTranspose", testWrapper(s.Suite, testGrumpkinTranspose))
}

func TestSuiteVecOps(t *testing.T) {
	suite.Run(t, new(VecOpsTestSuite))
}
