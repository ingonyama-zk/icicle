package cuda_runtime

type DeviceContext struct {
	/// Stream to use. Default value: 0.
	Stream *Stream // Assuming the type is provided by a CUDA binding crate

	/// Index of the currently used GPU. Default value: 0.
	DeviceId uint

	/// Mempool to use. Default value: 0.
	// TODO: use cuda_bindings.CudaMemPool as type
	Mempool uint // Assuming the type is provided by a CUDA binding crate
}

func GetDefaultDeviceContext() (DeviceContext, CudaError) {
	defaultStream, err := CreateStream()
	if err != CudaSuccess {
		return DeviceContext{}, err
	}

	return DeviceContext {
			&defaultStream,
			0,
			0,
	}, CudaSuccess
}