package merkletree

// #cgo LDFLAGS: -L/usr/local/lib  -licicle_hash -lstdc++ -Wl,-rpath=/usr/local/lib
// #cgo CFLAGS: -I./include/
// #include "merkletree.h"
import "C"
import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

type MerkleProofHandle = *C.struct_MerkleProof
type MerkleTreeHandle = *C.struct_MerkleTree

type MerkleProof struct {
	handle MerkleProofHandle
}

func CreateMerkleProof() (MerkleProof, runtime.EIcicleError) {
	h := C.icicle_merkle_proof_create()
	if h == nil {
		return MerkleProof{}, runtime.AllocationFailed
	}

	return MerkleProof{
		handle: h,
	}, runtime.Success
}

func (mp *MerkleProof) Delete() runtime.EIcicleError {
	if mp.handle == nil {
		return runtime.Success
	}

	err := C.icicle_merkle_proof_delete(mp.handle)
	icicleError := runtime.EIcicleError(err)
	if icicleError == runtime.Success {
		mp.handle = nil
	}
	return icicleError
}

func (mp *MerkleProof) IsPruned() bool {
	return (bool)(C.icicle_merkle_proof_is_pruned(mp.handle))
}

func GetMerkleProofPath[T any](mp *MerkleProof) []T {
	var size uint64
	cSize := (*C.uint64_t)(unsafe.Pointer(&size))

	pathPtr := C.icicle_merkle_proof_get_path(mp.handle, cSize)
	if pathPtr == nil {
		return []T{}
	} else {
		sizeOfElement := uint64(reflect.TypeFor[T]().Size())
		length := size / sizeOfElement
		return unsafe.Slice((*T)(unsafe.Pointer(pathPtr)), length)
	}
}

func GetMerkleProofLeaf[T any](mp *MerkleProof) ([]T, uint64) {
	var size uint64
	cSize := (*C.uint64_t)(unsafe.Pointer(&size))
	var leafIndex uint64
	cLeafIndex := (*C.uint64_t)(unsafe.Pointer(&leafIndex))

	leafPtr := C.icicle_merkle_proof_get_leaf(mp.handle, cSize, cLeafIndex)
	if leafPtr == nil {
		return []T{}, 0
	} else {
		sizeOfElement := uint64(reflect.TypeFor[T]().Size())
		length := size / sizeOfElement
		return unsafe.Slice((*T)(unsafe.Pointer(leafPtr)), length), leafIndex
	}
}

func GetMerkleProofRoot[T any](mp *MerkleProof) []T {
	var size uint64
	cSize := (*C.uint64_t)(unsafe.Pointer(&size))

	pathPtr := C.icicle_merkle_proof_get_path(mp.handle, cSize)
	if pathPtr == nil {
		return []T{}
	} else {
		sizeOfElement := uint64(reflect.TypeFor[T]().Size())
		length := size / sizeOfElement
		return unsafe.Slice((*T)(unsafe.Pointer(pathPtr)), length)
	}
}

type MerkleTree struct {
	handle MerkleTreeHandle
}

func CreateMerkleTree(
	layerHashers []hash.Hasher,
	leafElementSize,
	outputStoreMinLayer uint64,
) (MerkleTree, runtime.EIcicleError) {
	var layerHasherHandles []hash.HasherHandle
	for _, hasher := range layerHashers {
		layerHasherHandles = append(layerHasherHandles, hasher.GetHandle())
	}

	merkleTreeHandle := C.icicle_merkle_tree_create(
		(**C.struct_Hash)(unsafe.Pointer(&layerHasherHandles[0])),
		C.ulong(len(layerHasherHandles)),
		C.ulong(leafElementSize),
		C.ulong(outputStoreMinLayer),
	)

	if merkleTreeHandle == nil {
		return MerkleTree{}, runtime.UnknownError
	} else {
		return MerkleTree{
			handle: merkleTreeHandle,
		}, runtime.Success
	}
}

func (mt *MerkleTree) Delete() runtime.EIcicleError {
	if mt.handle == nil {
		return runtime.Success
	}

	err := C.icicle_merkle_tree_delete(mt.handle)
	icicleError := runtime.EIcicleError(err)
	if icicleError == runtime.Success {
		mt.handle = nil
	}
	return icicleError
}

func (mt *MerkleTree) Verify(mp *MerkleProof) (bool, runtime.EIcicleError) {
	isVerified := false
	err := C.icicle_merkle_tree_verify(mt.handle, mp.handle, (*C._Bool)(unsafe.Pointer(&isVerified)))
	icicleErr := runtime.EIcicleError(err)
	return isVerified, icicleErr
}

func BuildMerkleTree[T any](
	mt *MerkleTree,
	leaves core.HostOrDeviceSlice,
	cfg core.MerkleTreeConfig,
) runtime.EIcicleError {
	if mt.handle == nil {
		fmt.Println("The MerkleTree was not initalized; Initialize first using CreateMerkleTree")
		return runtime.InvalidArgument
	}

	leavesPtr := core.MerkleTreeCheck(leaves, &cfg)
	cLeaves := (*C.uint8_t)(leavesPtr)
	sizeOfElement := int(reflect.TypeFor[T]().Size())
	cLeavesSizeInBytes := (C.ulong)(leaves.Len() * sizeOfElement)
	cCfg := (*C.MerkleTreeConfig)(unsafe.Pointer(&cfg))
	__err := C.icicle_merkle_tree_build(mt.handle, cLeaves, cLeavesSizeInBytes, cCfg)
	return runtime.EIcicleError(__err)
}

func GetMerkleTreeRoot[T any](mt *MerkleTree) ([]T, runtime.EIcicleError) {
	if mt.handle == nil {
		fmt.Println("The MerkleTree was not initalized; Initialize first using CreateMerkleTree")
		return []T{}, runtime.InvalidArgument
	}

	var size uint64
	cSize := (*C.uint64_t)(unsafe.Pointer(&size))

	rootPtr := C.icicle_merkle_tree_get_root(mt.handle, cSize)
	if rootPtr == nil {
		return []T{}, runtime.UnknownError
	} else {
		sizeOfElement := uint64(reflect.TypeFor[T]().Size())
		length := size / sizeOfElement
		return unsafe.Slice((*T)(unsafe.Pointer(rootPtr)), length), runtime.Success
	}
}

func GetMerkleTreeProof[T any](
	mt *MerkleTree,
	leaves core.HostOrDeviceSlice,
	leafIndex uint64,
	prunedPath bool,
	cfg core.MerkleTreeConfig,
) (MerkleProof, runtime.EIcicleError) {
	if mt.handle == nil {
		fmt.Println("The MerkleTree was not initalized; Initialize first using CreateMerkleTree")
		return MerkleProof{}, runtime.InvalidArgument
	}

	proof, err := CreateMerkleProof()
	if err != runtime.Success {
		return MerkleProof{}, err
	}

	leavesPtr := core.MerkleTreeCheck(leaves, &cfg)
	cLeaves := (*C.uint8_t)(leavesPtr)
	sizeOfElement := int(reflect.TypeFor[T]().Size())
	cLeavesSizeInBytes := (C.ulong)(leaves.Len() * sizeOfElement)
	cLeafIndex := (C.uint64_t)(leafIndex)
	cPrunedPath := (C._Bool)(prunedPath)
	cCfg := (*C.MerkleTreeConfig)(unsafe.Pointer(&cfg))

	__err := C.icicle_merkle_tree_get_proof(
		mt.handle,
		cLeaves,
		cLeavesSizeInBytes,
		cLeafIndex,
		cPrunedPath,
		cCfg,
		proof.handle,
	)

	return proof, runtime.EIcicleError(__err)
}
