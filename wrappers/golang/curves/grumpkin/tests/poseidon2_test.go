package tests

import (
	"math"
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	grumpkin "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin/poseidon2"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/internal/test_helpers"
	merkletree "github.com/ingonyama-zk/icicle/v3/wrappers/golang/merkle-tree"
	"github.com/stretchr/testify/suite"
)

func testPoseidon2Hash(s *suite.Suite) {
	batch := 1 << 4
	domainTag := grumpkin.GenerateScalars(1)[0]
	for _, t := range []int{2, 3, 4, 8, 12, 16, 20, 24} {
		// TODO Danny add  8, 12, 16, 20, 24 for large fields once all is supported
		var largeField grumpkin.ScalarField
		if largeField.Size() > 4 && t > 4 {
			continue // TODO Danny remove this
		}
		for _, domainTag := range []*grumpkin.ScalarField{nil, &domainTag} {
			var inputsSize int
			if domainTag != nil {
				inputsSize = batch * (t - 1)
			} else {
				inputsSize = batch * t
			}

			inputs := grumpkin.GenerateScalars(inputsSize)

			// Set device to CPU
			test_helpers.ActivateReferenceDevice()
			outputsRef := make([]grumpkin.ScalarField, batch)
			poseidon2HasherRef, _ := poseidon2.NewHasher(uint64(t), domainTag)
			poseidon2HasherRef.Hash(
				core.HostSliceFromElements(inputs),
				core.HostSliceFromElements(outputsRef),
				core.GetDefaultHashConfig(),
			)

			poseidon2HasherRef.Delete()

			// Set device to main
			test_helpers.ActivateMainDevice()
			outputsMain := make([]grumpkin.ScalarField, batch)
			poseidon2HasherMain, _ := poseidon2.NewHasher(uint64(t), domainTag)
			poseidon2HasherMain.Hash(
				core.HostSliceFromElements(inputs),
				core.HostSliceFromElements(outputsMain),
				core.GetDefaultHashConfig(),
			)

			poseidon2HasherMain.Delete()

			s.Equal(outputsRef, outputsMain, "Poseidon2 hash outputs did not match")
		}
	}
}

// TODO uncomment once Poseidon2 sponge function is ready
// func testPoseidon2HashSponge(s *suite.Suite) {
// 	for _, t := range []int{2, 3, 4, 8, 12, 16, 20, 24} {
// 		inputs := grumpkin.GenerateScalars(t*8 - 2)
// 		var noDomainTag *grumpkin.ScalarField
//
// 		// Set device to CPU
// 		test_helpers.ActivateReferenceDevice()
// 		outputsRef := make([]grumpkin.ScalarField, 1)
// 		poseidon2HasherRef, _ := poseidon2.NewHasher(uint64(t), noDomainTag)
// 		err := poseidon2HasherRef.Hash(
// 			core.HostSliceFromElements(inputs),
// 			core.HostSliceFromElements(outputsRef),
// 			core.GetDefaultHashConfig(),
// 		)
//
// 		poseidonHasherRef.Delete()
//
// 		// Set device to main
// 		test_helpers.ActivateMainDevice()
// 		outputsMain := make([]grumpkin.ScalarField, 1)
// 		poseidonHasherMain, _ := poseidon.NewHasher(uint64(t), noDomainTag)
// 		poseidonHasherMain.Hash(
// 			core.HostSliceFromElements(inputs),
// 			core.HostSliceFromElements(outputsMain),
// 			core.GetDefaultHashConfig(),
// 		)
//
// 		poseidon2HasherMain.Delete()
//
// 		s.Equal(runtime.InvalidArgument, err)
// 	}
// }

func testPoseidon2HashTree(s *suite.Suite) {
	t := 4
	numLayers := 4
	numElements := int(math.Pow(4, float64(numLayers)))
	leaves := grumpkin.GenerateScalars(int(numElements))

	var noDomainTag *grumpkin.ScalarField
	hasher, _ := poseidon2.NewHasher(uint64(t), noDomainTag)
	layerHashers := make([]hash.Hasher, numLayers)
	for i := 0; i < numLayers; i++ {
		layerHashers[i] = hasher
	}

	var scalar grumpkin.ScalarField
	merkleTree, _ := merkletree.CreateMerkleTree(layerHashers, uint64(scalar.Size()), 0)
	merkletree.BuildMerkleTree[grumpkin.ScalarField](&merkleTree, core.HostSliceFromElements(leaves), core.GetDefaultMerkleTreeConfig())

	leafIndexToOpen := numElements >> 1
	merkleProof, _ := merkletree.GetMerkleTreeProof[grumpkin.ScalarField](&merkleTree, core.HostSliceFromElements(leaves), uint64(leafIndexToOpen), true /* prunedPath */, core.GetDefaultMerkleTreeConfig())

	// Examples of how to access different pieces of the proof
	_ /*root*/ = merkletree.GetMerkleProofRoot[grumpkin.ScalarField](&merkleProof)
	_ /*path*/ = merkletree.GetMerkleProofPath[grumpkin.ScalarField](&merkleProof)
	_ /*leaf*/, _ /*leafIndex*/ = merkletree.GetMerkleProofLeaf[grumpkin.ScalarField](&merkleProof)

	valid, _ := merkleTree.Verify(&merkleProof)
	s.True(valid)

	// TODO - add test for GPU when it is ready
}

type Poseidon2TestSuite struct {
	suite.Suite
}

func (s *Poseidon2TestSuite) TestPoseidon2() {
	s.Run("TestPoseidon2Hash", testWrapper(&s.Suite, testPoseidon2Hash))
	// s.Run("TestPoseidonHash2Sponge", testWrapper(&s.Suite, testPoseidonHash2Sponge))
	s.Run("TestPoseidon2HashTree", testWrapper(&s.Suite, testPoseidon2HashTree))
}

func TestSuitePoseidon2(t *testing.T) {
	suite.Run(t, new(Poseidon2TestSuite))
}
