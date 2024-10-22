package core

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"
)

type PaddingPolicy = int
const (
	NoPadding 			PaddingPolicy = iota    // No padding, assume input is correctly sized.
	ZeroPadding 														// Pad the input with zeroes to fit the expected input size.
	LastValuePadding   											// Pad the input by repeating the last value.
)

type MerkleTreeConfig struct {
	StreamHandle runtime.Stream
	areLeavesOnDevice bool 
	isTreeOnDevice bool
	IsAsync bool
	PaddingPolicy PaddingPolicy 
	Ext config_extension.ConfigExtensionHandler
}

func GetDefaultMerkleTreeConfig() MerkleTreeConfig {
	return MerkleTreeConfig {
		StreamHandle: nil,
		areLeavesOnDevice: false,
		isTreeOnDevice: false,
		IsAsync: false,
		PaddingPolicy: NoPadding,
		Ext: nil,
	}
}

func MerkleTreeCheck(leaves HostOrDeviceSlice, cfg *MerkleTreeConfig) unsafe.Pointer {
	if leaves.IsOnDevice() {
		leaves.(DeviceSlice).CheckDevice()	
		cfg.areLeavesOnDevice = true
	}

	return leaves.AsUnsafePointer()
}
