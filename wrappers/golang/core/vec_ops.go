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
	Sum
	Product
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
	Ext     config_extension.ConfigExtensionHandler
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
		nil,   // Ext
	}

	return config
}

func VecReduceOpCheck(a, out HostOrDeviceSlice, cfg *VecOpsConfig) (unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int) {
	aLen, outLen := a.Len(), out.Len()
	
	if outLen < 1 {
		errorString := fmt.Sprintf(
			"out length %d < 1",
			outLen,
		)
		panic(errorString)
	}

	if a.IsOnDevice() {
		a.(DeviceSlice).CheckDevice()
	}
	if out.IsOnDevice() {
		out.(DeviceSlice).CheckDevice()
	}

	cfg.isAOnDevice = a.IsOnDevice()
	cfg.isResultOnDevice = out.IsOnDevice()

	return a.AsUnsafePointer(), out.AsUnsafePointer(), unsafe.Pointer(cfg), a.Len()
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
