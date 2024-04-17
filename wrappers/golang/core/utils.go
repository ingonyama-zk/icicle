package core

import (
	"encoding/binary"
)

func ConvertUint32ArrToUint64Arr(arr32 []uint32) []uint64 {
	arr64 := make([]uint64, len(arr32)/2)
	for i := 0; i < len(arr32); i += 2 {
		arr64[i/2] = (uint64(arr32[i]) << 32) | uint64(arr32[i+1])
	}
	return arr64
}

func ConvertUint64ArrToUint32Arr(arr64 []uint64) []uint32 {
	arr32 := make([]uint32, len(arr64)*2)
	for i, v := range arr64 {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, v)

		arr32[i*2] = binary.LittleEndian.Uint32(b[0:4])
		arr32[i*2+1] = binary.LittleEndian.Uint32(b[4:8])
	}

	return arr32
}