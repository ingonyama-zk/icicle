# Keccak

TODO update for V3

## Keccak Example

```go
package main

import (
	"encoding/hex"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/hash/keccak"
)

func createHostSliceFromHexString(hexString string) core.HostSlice[uint8] {
	byteArray, err := hex.DecodeString(hexString)
	if err != nil {
		panic("Not a hex string")
	}
	return core.HostSliceFromElements([]uint8(byteArray))
}

func main() {
	input := createHostSliceFromHexString("1725b6")
	outHost256 := make(core.HostSlice[uint8], 32)

	cfg := keccak.GetDefaultHashConfig()
	e := keccak.Keccak256(input, int32(input.Len()), 1, outHost256, &cfg)
	if e.CudaErrorCode != cr.CudaSuccess {
		panic("Keccak256 hashing failed")
	}

	outHost512 := make(core.HostSlice[uint8], 64)
	e = keccak.Keccak512(input, int32(input.Len()), 1, outHost512, &cfg)
	if e.CudaErrorCode != cr.CudaSuccess {
		panic("Keccak512 hashing failed")
	}

    numberOfBlocks := 3
	outHostBatch256 := make(core.HostSlice[uint8], 32*numberOfBlocks)
	e = keccak.Keccak256(input, int32(input.Len()/numberOfBlocks), int32(numberOfBlocks), outHostBatch256, &cfg)
	if e.CudaErrorCode != cr.CudaSuccess {
		panic("Keccak256 batch hashing failed")
	}
}
```

## Keccak Methods

```go
func Keccak256(input core.HostOrDeviceSlice, inputBlockSize, numberOfBlocks int32, output core.HostOrDeviceSlice, config *HashConfig) core.IcicleError
func Keccak512(input core.HostOrDeviceSlice, inputBlockSize, numberOfBlocks int32, output core.HostOrDeviceSlice, config *HashConfig) core.IcicleError
```

### Parameters

- **`input`**: A slice containing the input data for the Keccak256 hash function. It can reside in either host memory or device memory.
- **`inputBlockSize`**: An integer specifying the size of the input data for a single hash.
- **`numberOfBlocks`**: An integer specifying the number of results in the hash batch.
- **`output`**: A slice where the resulting hash will be stored. This slice can be in host or device memory.
- **`config`**: A pointer to a `HashConfig` object, which contains various configuration options for the Keccak256 operation.

### Return Value

- **`CudaError`**: Returns a CUDA error code indicating the success or failure of the Keccak256/Keccak512 operation.

## HashConfig

The `HashConfig` structure holds configuration parameters for the Keccak256/Keccak512 operation, allowing customization of its behavior to optimize performance based on the specifics of the operation or the underlying hardware.

```go
type HashConfig struct {
	Ctx                cr.DeviceContext
	areInputsOnDevice  bool
	areOutputsOnDevice bool
	IsAsync            bool
}
```

### Fields

- **`Ctx`**: Device context containing details like device id and stream.
- **`areInputsOnDevice`**: Indicates if input data is located on the device.
- **`areOutputsOnDevice`**: Indicates if output hash is stored on the device.
- **`IsAsync`**: If true, runs the Keccak256/Keccak512 operation asynchronously.

### Default Configuration

Use `GetDefaultHashConfig` to obtain a default configuration, which can then be customized as needed.

```go
func GetDefaultHashConfig() HashConfig
```