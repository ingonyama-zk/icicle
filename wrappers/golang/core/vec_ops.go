package core

import (
	"fmt"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime/config_extension"
)

type VecOps int

const (
	Sub VecOps = iota
	Add
	Mul
)

type VecOpsConfig struct {
	/// Specifies the stream (queue) to use for async execution.
	StreamHandle runtime.Stream
	/// True if `a` is on device and false if it is not. Default value: false.
	isAOnDevice bool
	/// True if `b` is on device and false if it is not. Default value: false.
	isBOnDevice bool
	/// If true, output is preserved on device, otherwise on host. Default value: false.
	isResultOnDevice bool
	/// Whether to run the vector operations asynchronously. If set to `true`, the function will be
	/// non-blocking and you'll need to synchronize it explicitly by calling
	/// `SynchronizeStream`. If set to false, the function will block the current CPU thread.
	IsAsync bool
	/// Number of vectors (or operations) to process in a batch.
	/// Each vector operation will be performed independently on each batch element.
	/// Default value: 1.
	BatchSize int32
	/// True if the batched vectors are stored as columns in a 2D array (i.e., the vectors are
	/// strided in memory as columns of a matrix). If false, the batched vectors are stored
	/// contiguously in memory (e.g., as rows or in a flat array). Default value: false.
	ColumnsBatch bool
	Ext          config_extension.ConfigExtensionHandler
}

/**
 * A function that returns the default value of [VecOpsConfig](@ref VecOpsConfig).
 * @return Default value of [VecOpsConfig](@ref VecOpsConfig).
 */
func DefaultVecOpsConfig() VecOpsConfig {
	config := VecOpsConfig{
		nil,   // StreamHandle
		false, // isAOnDevice
		false, // isBOnDevice
		false, // isResultOnDevice
		false, // IsAsync
		1,     // BatchSize
		false, // ColumnsBatch
		nil,   // Ext
	}

	return config
}

func VecOpCheck(a, b, out HostOrDeviceSlice, cfg *VecOpsConfig) (unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int) {
	aLen, bLen, outLen := a.Len(), b.Len(), out.Len()
	if aLen != bLen {
		errorString := fmt.Sprintf(
			"a and b vector lengths %d; %d are not equal",
			aLen,
			bLen,
		)
		panic(errorString)
	}
	if aLen != outLen {
		errorString := fmt.Sprintf(
			"a and out vector lengths %d; %d are not equal",
			aLen,
			outLen,
		)
		panic(errorString)
	}

	if a.IsOnDevice() {
		a.(DeviceSlice).CheckDevice()
	}
	if b.IsOnDevice() {
		b.(DeviceSlice).CheckDevice()
	}
	if out.IsOnDevice() {
		out.(DeviceSlice).CheckDevice()
	}

	cfg.isAOnDevice = a.IsOnDevice()
	cfg.isBOnDevice = b.IsOnDevice()
	cfg.isResultOnDevice = out.IsOnDevice()

	return a.AsUnsafePointer(), b.AsUnsafePointer(), out.AsUnsafePointer(), unsafe.Pointer(cfg), a.Len()
}

func TransposeCheck(in, out HostOrDeviceSlice, onDevice bool) {
	inLen, outLen := in.Len(), out.Len()

	if inLen != outLen {
		errorString := fmt.Sprintf(
			"in and out vector lengths %d; %d are not equal",
			inLen,
			outLen,
		)
		panic(errorString)
	}
	if (onDevice != in.IsOnDevice()) || (onDevice != out.IsOnDevice()) {
		errorString := fmt.Sprintf(
			"onDevice is set to %t, but in.IsOnDevice():%t and out.IsOnDevice():%t",
			onDevice,
			in.IsOnDevice(),
			out.IsOnDevice(),
		)
		panic(errorString)
	}
	if onDevice {
		in.(DeviceSlice).CheckDevice()
		out.(DeviceSlice).CheckDevice()
	}
}
