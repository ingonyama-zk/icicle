package core

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDefaultMerkleTreeConfig(t *testing.T) {
	actual := GetDefaultMerkleTreeConfig()

	expected := MerkleTreeConfig{
		StreamHandle:       nil,
		areLeavesOnDevice:  false,
		isTreeOnDevice:  		false,
		IsAsync:            false,
		PaddingPolicy: 			NoPadding,
		Ext:                nil,
	}

	assert.EqualValues(t, expected, actual)
}

func TestCheckMerkleTree(t *testing.T) {
	cfg := GetDefaultMerkleTreeConfig()
	input := make([]byte, 512)
	
	inputHost := HostSliceFromElements(input)
	assert.NotPanics(t, func() { MerkleTreeCheck(inputHost, &cfg) })
	assert.False(t, cfg.areLeavesOnDevice)
	
	var d_input DeviceSlice
	inputHost.CopyToDevice(&d_input, true)
	assert.NotPanics(t, func() { MerkleTreeCheck(d_input, &cfg) })
	assert.True(t, cfg.areLeavesOnDevice)
}
