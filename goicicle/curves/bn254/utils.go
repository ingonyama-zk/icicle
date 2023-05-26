package bn254

// Function to convert [8]uint32 to [4]uint64
func convertUint32ArrToUint64Arr(arr32 [8]uint32) [4]uint64 {
	var arr64 [4]uint64
	for i := 0; i < len(arr32); i += 2 {
		arr64[i/2] = (uint64(arr32[i]) << 32) | uint64(arr32[i+1])
	}
	return arr64
}

// Function to convert [4]uint64 to [8]uint32
func convertUint64ArrToUint32Arr(arr64 [4]uint64) [8]uint32 {
	var arr32 [8]uint32
	for i, v := range arr64 {
		arr32[i*2] = uint32(v >> 32)
		arr32[i*2+1] = uint32(v & 0xFFFFFFFF)
	}
	return arr32
}
