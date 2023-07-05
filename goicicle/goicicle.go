package goicicle

// This file implements CUDA driver context management

//#include <cuda.h>
import "C"

// Version returns the version of the CUDA driver
func Version() int {
	var v C.int
	if err := C.cuDriverGetVersion(&v); err != 0 {
		return -1
	}
	return int(v)
}
