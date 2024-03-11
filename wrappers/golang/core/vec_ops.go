package core

import (
	"fmt"
	"unsafe"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

type VecOps int

const (
	Sub VecOps = iota
	Add
	Mul
)

type VecOpsConfig struct {
	/*Details related to the device such as its id and stream. */
	Ctx cr.DeviceContext
	/* True if `a` is on device and false if it is not. Default value: false. */
	isAOnDevice bool
	/* True if `b` is on device and false if it is not. Default value: false. */
	isBOnDevice bool
	/* If true, output is preserved on device, otherwise on host. Default value: false. */
	isResultOnDevice bool
	/* True if `result` vector should be in Montgomery form and false otherwise. Default value: false. */
	IsResultMontgomeryForm bool
	/* Whether to run the vector operations asynchronously. If set to `true`, the function will be
	*  non-blocking and you'll need to synchronize it explicitly by calling
	*  `SynchronizeStream`. If set to false, the function will block the current CPU thread. */
	IsAsync bool
}

/**
 * A function that returns the default value of [VecOpsConfig](@ref VecOpsConfig).
 * @return Default value of [VecOpsConfig](@ref VecOpsConfig).
 */
func DefaultVecOpsConfig() VecOpsConfig {
	ctx, _ := cr.GetDefaultDeviceContext()
	config := VecOpsConfig{
		ctx,   // ctx
		false, // isAOnDevice
		false, // isBOnDevice
		false, // isResultOnDevice
		false, // IsResultMontgomeryForm
		false, // IsAsync
	}

	return config
}

func VecOpCheck[S HostSliceInterface](a, b, out HostOrDeviceSlice, cfg *VecOpsConfig) (unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int) {
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

	cfg.isAOnDevice = a.IsOnDevice()
	cfg.isBOnDevice = b.IsOnDevice()
	cfg.isResultOnDevice = out.IsOnDevice()
	
	var aPointer, bPointer, outPointer unsafe.Pointer
	if a.IsOnDevice() {
		aPointer = a.(DeviceSlice).AsPointer()
	} else {
		aPointer = unsafe.Pointer(&a.(HostSlice[S])[0])
	}

	if b.IsOnDevice() {
		bPointer = b.(DeviceSlice).AsPointer()
	} else {
		bPointer = unsafe.Pointer(&b.(HostSlice[S])[0])
	}

	if out.IsOnDevice() {
		outPointer = out.(DeviceSlice).AsPointer()
	} else {
		outPointer = unsafe.Pointer(&out.(HostSlice[S])[0])
	}

	cfgPointer := unsafe.Pointer(cfg)
	size := a.Len()

	return aPointer, bPointer, outPointer, cfgPointer, size
}
