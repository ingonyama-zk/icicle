//go:build !icicle_exclude_all || poseidon

package tests

import (
	"math"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	stark252 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/stark252"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/fields/stark252/poseidon"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	merkletree "github.com/ingonyama-zk/icicle/v3/wrappers/golang/merkle-tree"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/stretchr/testify/suite"
)

func testPoseidonHash(s *suite.Suite) {
	batch := 1 << 4
	domainTag := stark252.GenerateScalars(1)[0]
	for _, t := range []int{3, 5, 9, 12} {
		for _, domainTag := range []*stark252.ScalarField{nil, &domainTag} {
			var inputsSize int
			if domainTag != nil {
				inputsSize = batch * (t - 1)
			} else {
				inputsSize = batch * t
			}

			inputs := stark252.GenerateScalars(inputsSize)

			// Set device to CPU
			test_helpers.ActivateReferenceDevice()
			outputsRef := make([]stark252.ScalarField, batch)
			poseidonHasherRef, _ := poseidon.NewHasher(uint64(t), domainTag)
			poseidonHasherRef.Hash(
				core.HostSliceFromElements(inputs),
				core.HostSliceFromElements(outputsRef),
				core.GetDefaultHashConfig(),
			)

			poseidonHasherRef.Delete()

			// Set device to main
			test_helpers.ActivateMainDevice()
			outputsMain := make([]stark252.ScalarField, batch)
			poseidonHasherMain, _ := poseidon.NewHasher(uint64(t), domainTag)
			poseidonHasherMain.Hash(
				core.HostSliceFromElements(inputs),
				core.HostSliceFromElements(outputsMain),
				core.GetDefaultHashConfig(),
			)

			poseidonHasherMain.Delete()

			s.Equal(outputsRef, outputsMain, "Poseidon hash outputs did not match")
		}
	}
}

func testPoseidonHashSponge(s *suite.Suite) {
	for _, t := range []int{3, 5, 9, 12} {
		inputs := stark252.GenerateScalars(t*8 - 2)
		var noDomainTag *stark252.ScalarField

		// Set device to CPU
		test_helpers.ActivateReferenceDevice()
		outputsRef := make([]stark252.ScalarField, 1)
		poseidonHasherRef, _ := poseidon.NewHasher(uint64(t), noDomainTag)
		err := poseidonHasherRef.Hash(
			core.HostSliceFromElements(inputs),
			core.HostSliceFromElements(outputsRef),
			core.GetDefaultHashConfig(),
		)

		poseidonHasherRef.Delete()

		// Set device to main
		test_helpers.ActivateMainDevice()
		outputsMain := make([]stark252.ScalarField, 1)
		poseidonHasherMain, _ := poseidon.NewHasher(uint64(t), noDomainTag)
		poseidonHasherMain.Hash(
			core.HostSliceFromElements(inputs),
			core.HostSliceFromElements(outputsMain),
			core.GetDefaultHashConfig(),
		)

		poseidonHasherMain.Delete()

		s.Equal(runtime.InvalidArgument, err)
	}
}

func testPoseidonHashTree(s *suite.Suite) {
	t := 9
	numLayers := 4
	numElements := int(math.Pow(9, float64(numLayers)))
	leaves := stark252.GenerateScalars(int(numElements))

	var noDomainTag *stark252.ScalarField
	hasher, _ := poseidon.NewHasher(uint64(t), noDomainTag)
	layerHashers := make([]hash.Hasher, numLayers)
	for i := 0; i < numLayers; i++ {
		layerHashers[i] = hasher
	}

	var scalar stark252.ScalarField
	merkleTree, _ := merkletree.CreateMerkleTree(layerHashers, uint64(scalar.Size()), 0)
	merkletree.BuildMerkleTree[stark252.ScalarField](&merkleTree, core.HostSliceFromElements(leaves), core.GetDefaultMerkleTreeConfig())

	leafIndexToOpen := numElements >> 1
	merkleProof, _ := merkletree.GetMerkleTreeProof[stark252.ScalarField](&merkleTree, core.HostSliceFromElements(leaves), uint64(leafIndexToOpen), true /* prunedPath */, core.GetDefaultMerkleTreeConfig())

	// Examples of how to access different pieces of the proof
	_ /*root*/ = merkletree.GetMerkleProofRoot[stark252.ScalarField](&merkleProof)
	_ /*path*/ = merkletree.GetMerkleProofPath[stark252.ScalarField](&merkleProof)
	_ /*leaf*/, _ /*leafIndex*/ = merkletree.GetMerkleProofLeaf[stark252.ScalarField](&merkleProof)

	valid, _ := merkleTree.Verify(&merkleProof)
	s.True(valid)

	// TODO - add test for GPU when it is ready
}

type PoseidonTestSuite struct {
	suite.Suite
}

func (s *PoseidonTestSuite) TestPoseidon() {
	s.Run("TestPoseidonHash", test_helpers.TestWrapper(&s.Suite, testPoseidonHash))
	s.Run("TestPoseidonHashSponge", test_helpers.TestWrapper(&s.Suite, testPoseidonHashSponge))
	s.Run("TestPoseidonHashTree", test_helpers.TestWrapper(&s.Suite, testPoseidonHashTree))
}

func TestSuitePoseidon(t *testing.T) {
	suite.Run(t, new(PoseidonTestSuite))
}
