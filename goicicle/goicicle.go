package goicicle

// This file implements CUDA driver context management

//#include <cuda.h>
import "C"

// Version returns the version of the CUDA driver
func Version() int {
	var v C.int
	if err := result(C.cuDriverGetVersion(&v)); err != nil {
		return -1
	}
	return int(v)
}
