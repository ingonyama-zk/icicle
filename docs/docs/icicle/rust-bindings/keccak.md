# Keccak

TODO update for V3

## Keccak Example

```rust
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use icicle_hash::keccak::{keccak256, HashConfig};
use rand::{self, Rng};

fn main() {
    let mut rng = rand::thread_rng();
    let initial_data: Vec<u8> = (0..120).map(|_| rng.gen::<u8>()).collect();
    println!("initial data: {}", hex::encode(&initial_data));
    let input = HostSlice::<u8>::from_slice(initial_data.as_slice());
    let mut output = DeviceVec::<u8>::cuda_malloc(32).unwrap();

    let mut config = HashConfig::default();
    keccak256(input, initial_data.len() as i32, 1, &mut output[..], &mut config).expect("Failed to execute keccak256 hashing");

    let mut output_host = vec![0_u8; 32];
    output.copy_to_host(HostSlice::from_mut_slice(&mut output_host[..])).unwrap();

    println!("keccak256 result: {}", hex::encode(&output_host));
}
```

## Keccak Methods

```rust
pub fn keccak256(
    input: &(impl HostOrDeviceSlice<u8> + ?Sized),
    input_block_size: i32,
    number_of_blocks: i32,
    output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
    config: &mut HashConfig,
) -> IcicleResult<()>

pub fn keccak512(
    input: &(impl HostOrDeviceSlice<u8> + ?Sized),
    input_block_size: i32,
    number_of_blocks: i32,
    output: &mut (impl HostOrDeviceSlice<u8> + ?Sized),
    config: &mut HashConfig,
) -> IcicleResult<()> 
```

### Parameters

- **`input`**: A slice containing the input data for the Keccak256 hash function. It can reside in either host memory or device memory.
- **`input_block_size`**: An integer specifying the size of the input data for a single hash.
- **`number_of_blocks`**: An integer specifying the number of results in the hash batch.
- **`output`**: A slice where the resulting hash will be stored. This slice can be in host or device memory.
- **`config`**: A pointer to a `HashConfig` object, which contains various configuration options for the Keccak256 operation.

### Return Value

- **`IcicleResult`**: Returns a CUDA error code indicating the success or failure of the Keccak256/Keccak512 operation.

## HashConfig

The `HashConfig` structure holds configuration parameters for the Keccak256/Keccak512 operation, allowing customization of its behavior to optimize performance based on the specifics of the operation or the underlying hardware.

```rust
pub struct HashConfig<'a> {
    pub ctx: DeviceContext<'a>,
    pub are_inputs_on_device: bool,
    pub are_outputs_on_device: bool,
    pub is_async: bool,
}
```

### Fields

- **`ctx`**: Device context containing details like device id and stream.
- **`are_inputs_on_device`**: Indicates if input data is located on the device.
- **`are_outputs_on_device`**: Indicates if output hash is stored on the device.
- **`is_async`**: If true, runs the Keccak256/Keccak512 operation asynchronously.

### Usage

Example initialization with default settings:

```rust
let default_config = HashConfig::default();
```

Customizing the configuration:

```rust
let custom_config = NTTConfig {
    ctx: custom_device_context,
    are_inputs_on_device: true,
    are_outputs_on_device: true,
    is_async: false,
};
```