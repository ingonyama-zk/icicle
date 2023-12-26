package core

import (
	"local/hello/icicle/wrappers/golang/cuda_runtime"
)

type IcicleErrorCode int 

const (
	IcicleSuccess 					IcicleErrorCode = 0
	InvalidArgument 				IcicleErrorCode = 1
	MemoryAllocationError 	IcicleErrorCode = 2
	InternalCudaError				IcicleErrorCode = 199999999
	UndefinedError 					IcicleErrorCode = 999999999
)

type IcicleError struct {
	IcicleErrorCode IcicleErrorCode
	CudaErrorCode	cuda_runtime.CudaError
	reason string
}

func FromCudaError(error cuda_runtime.CudaError) (err IcicleError) {
	switch (error) {
	case cuda_runtime.CudaSuccess:
		err.IcicleErrorCode = IcicleSuccess
	default:
		err.IcicleErrorCode = InternalCudaError
	}

	err.CudaErrorCode = error
	err.reason = "Runtime CUDA error."

	return err
}

func FromCodeAndReason(code IcicleErrorCode, reason string) IcicleError {
	return IcicleError {
		IcicleErrorCode: code,
		reason: reason,
		CudaErrorCode: cuda_runtime.CudaErrorUnknown,
	}
}