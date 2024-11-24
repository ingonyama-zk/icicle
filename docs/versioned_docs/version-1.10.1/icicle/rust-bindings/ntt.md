# NTT

### Supported curves

`bls12-377`, `bls12-381`, `bn-254`, `bw6-761`

## Example 

```rust
use icicle_bn254::curve::{ScalarCfg, ScalarField};
use icicle_core::{ntt::{self, NTT}, traits::GenerateRandom};
use icicle_cuda_runtime::{device_context::DeviceContext, memory::HostOrDeviceSlice, stream::CudaStream};

fn main() {
    let size = 1 << 12; // Define the size of your input, e.g., 2^10

    let icicle_omega = <Bn254Fr as FftField>::get_root_of_unity(
        size.try_into()
            .unwrap(),
    )

    // Generate random inputs
    println!("Generating random inputs...");
    let scalars = HostOrDeviceSlice::Host(ScalarCfg::generate_random(size));

    // Allocate memory on CUDA device for NTT results
    let mut ntt_results: HostOrDeviceSlice<'_, ScalarField> = HostOrDeviceSlice::cuda_malloc(size).expect("Failed to allocate CUDA memory");

    // Create a CUDA stream
    let stream = CudaStream::create().expect("Failed to create CUDA stream");
    let ctx = DeviceContext::default(); // Assuming default device context
    ScalarCfg::initialize_domain(ScalarField::from_ark(icicle_omega), &ctx).unwrap();

    // Configure NTT
    let mut cfg = ntt::NTTConfig::default();
    cfg.ctx.stream = &stream;
    cfg.is_async = true; // Set to true for asynchronous execution

    // Execute NTT on device
    println!("Executing NTT on device...");
    ntt::ntt(&scalars, ntt::NTTDir::kForward, &cfg, &mut ntt_results).expect("Failed to execute NTT");

    // Synchronize CUDA stream to ensure completion
    stream.synchronize().expect("Failed to synchronize CUDA stream");

    // Optionally, move results to host for further processing or verification
    println!("NTT execution complete.");
}
```

## NTT API overview

```rust
pub fn ntt<F>(
    input: &HostOrDeviceSlice<F>,
    dir: NTTDir,
    cfg: &NTTConfig<F>,
    output: &mut HostOrDeviceSlice<F>,
) -> IcicleResult<()>
```

`ntt:ntt` expects:

`input` - buffer to read the inputs of the NTT from. <br/>
`dir` - whether to compute forward or inverse NTT. <br/>
`cfg` - config used to specify extra arguments of the NTT. <br/>
`output` - buffer to write the NTT outputs into. Must be of the same  size as input.

The `input` and `output` buffers can be on device or on host. Being on host means that they will be transferred to device during runtime.


### NTT Config

```rust
pub struct NTTConfig<'a, S> {
    pub ctx: DeviceContext<'a>,
    pub coset_gen: S,
    pub batch_size: i32,
    pub columns_batch: bool,
    pub ordering: Ordering,
    are_inputs_on_device: bool,    
    are_outputs_on_device: bool,
    pub is_async: bool,
    pub ntt_algorithm: NttAlgorithm,
}
```

The `NTTConfig` struct is a configuration object used to specify parameters for an NTT instance.

#### Fields

- **`ctx: DeviceContext<'a>`**: Specifies the device context, including the device ID and the stream ID.

- **`coset_gen: S`**: Defines the coset generator used for coset (i)NTTs. By default, this is set to `S::one()`, indicating that no coset is being used.

- **`batch_size: i32`**: Determines the number of NTTs to compute in a single batch. The default value is 1, meaning that operations are performed on individual inputs without batching. Batch processing can significantly improve performance by leveraging parallelism in GPU computations.

- **`columns_batch`**: If true the function will compute the NTTs over the columns of the input matrix and not over the rows. Defaults to `false`.

- **`ordering: Ordering`**: Controls the ordering of inputs and outputs for the NTT operation. This field can be used to specify decimation strategies (in time or in frequency) and the type of butterfly algorithm (Cooley-Tukey or Gentleman-Sande). The ordering is crucial for compatibility with various algorithmic approaches and can impact the efficiency of the NTT.

- **`are_inputs_on_device: bool`**: Indicates whether the input data has been preloaded on the device memory. If `false` inputs will be copied from host to device.

- **`are_outputs_on_device: bool`**: Indicates whether the output data is preloaded in device memory. If `false` outputs will be copied from host to device. If the inputs and outputs are the same pointer NTT will be computed in place.

- **`is_async: bool`**: Specifies whether the NTT operation should be performed asynchronously. When set to `true`, the NTT function will not block the CPU, allowing other operations to proceed concurrently. Asynchronous execution requires careful synchronization to ensure data integrity and correctness.

- **`ntt_algorithm: NttAlgorithm`**: Can be one of `Auto`, `Radix2`, `MixedRadix`.
`Auto` will select `Radix 2` or `Mixed Radix` algorithm based on heuristics.
`Radix2` and `MixedRadix` will force the use of an algorithm regardless of the input size or other considerations. You should use one of these options when you know for sure that you want to 


#### Usage

Example initialization with default settings:

```rust
let default_config = NTTConfig::default();
```

Customizing the configuration:

```rust
let custom_config = NTTConfig {
    ctx: custom_device_context,
    coset_gen: my_coset_generator,
    batch_size: 10,
    columns_batch: false,
    ordering: Ordering::kRN,
    are_inputs_on_device: true,
    are_outputs_on_device: true,
    is_async: false,
    ntt_algorithm: NttAlgorithm::MixedRadix,
};
```


### Modes

NTT supports two different modes `Batch NTT` and `Single NTT`

You may toggle between single and batch NTT by simply configure `batch_size` to be larger then 1 in your `NTTConfig`.

```rust
let mut cfg = ntt::get_default_ntt_config::<ScalarField>();
cfg.batch_size = 10 // your ntt using this config will run in batch mode.
```

`batch_size=1` would keep our NTT in single NTT mode.

Deciding weather to use `batch NTT` vs `single NTT` is highly dependent on your application and use case.

### Initializing the NTT Domain

Before performing NTT operations, its necessary to initialize the NTT domain, It only needs to be called once per GPU since the twiddles are cached.

```rust
ScalarCfg::initialize_domain(ScalarField::from_ark(icicle_omega), &ctx).unwrap();
```

### `initialize_domain`

```rust
pub fn initialize_domain<F>(primitive_root: F, ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>;
```

#### Parameters

- **`primitive_root`**: The primitive root of unity, chosen based on the maximum NTT size required for the computations. It must be of an order that is a power of two. This root is used to generate twiddle factors that are essential for the NTT operations.

- **`ctx`**: A reference to a `DeviceContext` specifying which device and stream the computation should be executed on.

#### Returns

- **`IcicleResult<()>`**: Will return an error if the operation fails.

### `initialize_domain_fast_twiddles_mode`

Similar to `initialize_domain`, `initialize_domain_fast_twiddles_mode` is a faster implementation and can be used for larger NTTs.

```rust
pub fn initialize_domain_fast_twiddles_mode<F>(primitive_root: F, ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>;
```

#### Parameters

- **`primitive_root`**: The primitive root of unity, chosen based on the maximum NTT size required for the computations. It must be of an order that is a power of two. This root is used to generate twiddle factors that are essential for the NTT operations.

- **`ctx`**: A reference to a `DeviceContext` specifying which device and stream the computation should be executed on.

#### Returns

- **`IcicleResult<()>`**: Will return an error if the operation fails.
