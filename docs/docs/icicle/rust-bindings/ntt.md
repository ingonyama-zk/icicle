# NTT

## NTT API overview

```rust
pub fn ntt<F>(
    input: &HostOrDeviceSlice<F>,
    dir: NTTDir,
    cfg: &NTTConfig<F>,
    output: &mut HostOrDeviceSlice<F>,
) -> eIcicleResult<()>
```

`ntt:ntt` expects:

- **`input`** - buffer to read the inputs of the NTT from.
- **`dir`** - whether to compute forward or inverse NTT.
- **`cfg`** - config used to specify extra arguments of the NTT.
- **`output`** - buffer to write the NTT outputs into. Must be of the same  size as input.

The `input` and `output` buffers can be on device or on host. Being on host means that they will be transferred to device during runtime.

### NTT Config

```rust
pub struct NTTConfig<S> {
    pub stream_handle: IcicleStreamHandle,
    pub coset_gen: S,
    pub batch_size: i32,
    pub columns_batch: bool,
    pub ordering: Ordering,
    pub are_inputs_on_device: bool,
    pub are_outputs_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension,
}
```

The `NTTConfig` struct is a configuration object used to specify parameters for an NTT instance.

#### Fields

- **`stream_handle: IcicleStreamHandle`**: Specifies the stream (queue) to use for async execution

- **`coset_gen: S`**: Defines the coset generator used for coset (i)NTTs. By default, this is set to `S::one()`, indicating that no coset is being used.

- **`batch_size: i32`**: Determines the number of NTTs to compute in a single batch. The default value is 1, meaning that operations are performed on individual inputs without batching. Batch processing can significantly improve performance by leveraging parallelism in GPU computations.

- **`columns_batch`**: If true the function will compute the NTTs over the columns of the input matrix and not over the rows. Defaults to `false`.

- **`ordering: Ordering`**: Controls the ordering of inputs and outputs for the NTT operation. This field can be used to specify decimation strategies (in time or in frequency) and the type of butterfly algorithm (Cooley-Tukey or Gentleman-Sande). The ordering is crucial for compatibility with various algorithmic approaches and can impact the efficiency of the NTT.

- **`are_inputs_on_device: bool`**: Indicates whether the input data has been preloaded on the device memory. If `false` inputs will be copied from host to device.

- **`are_outputs_on_device: bool`**: Indicates whether the output data is preloaded in device memory. If `false` outputs will be copied from host to device. If the inputs and outputs are the same pointer NTT will be computed in place.

- **`is_async: bool`**: Specifies whether the NTT operation should be performed asynchronously. When set to `true`, the NTT function will not block the CPU, allowing other operations to proceed concurrently. Asynchronous execution requires careful synchronization to ensure data integrity and correctness.
- **`ext: ConfigExtension`**: extended configuration for backend.

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


TODO update for V3

#### Example - TODO update for V3

```rust
// Setting Bn254 points and scalars
println!("Generating random inputs on host for bn254...");
let scalars = Bn254ScalarCfg::generate_random(size);
let mut ntt_results = DeviceVec::<Bn254ScalarField>::device_malloc(size).unwrap();

// constructin NTT domain
initialize_domain(
    ntt::get_root_of_unity::<Bn254ScalarField>(
        size.try_into()
            .unwrap(),
    ),
    &ntt::NTTInitDomainConfig::default(),
)
.unwrap();

// Using default config
let cfg = ntt::NTTConfig::<Bn254ScalarField>::default();

// Computing NTT
ntt::ntt(
    HostSlice::from_slice(&scalars),
    ntt::NTTDir::kForward,
    &cfg,
    &mut ntt_results[..],
)
.unwrap();
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
ScalarCfg::initialize_domain(ScalarField::from_ark(icicle_omega), &ctx, true).unwrap();
```

### `initialize_domain`

```rust
pub fn initialize_domain<F>(primitive_root: F, ctx: &DeviceContext, fast_twiddles: bool) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>;
```

#### Parameters

- **`primitive_root`**: The primitive root of unity, chosen based on the maximum NTT size required for the computations. It must be of an order that is a power of two. This root is used to generate twiddle factors that are essential for the NTT operations.

- **`ctx`**: A reference to a `DeviceContext` specifying which device and stream the computation should be executed on.

#### Returns

- **`IcicleResult<()>`**: Will return an error if the operation fails.

#### Parameters

- **`primitive_root`**: The primitive root of unity, chosen based on the maximum NTT size required for the computations. It must be of an order that is a power of two. This root is used to generate twiddle factors that are essential for the NTT operations.

- **`ctx`**: A reference to a `DeviceContext` specifying which device and stream the computation should be executed on.

#### Returns

- **`IcicleResult<()>`**: Will return an error if the operation fails.

### Releasing the domain

The `release_domain` function is responsible for releasing the resources associated with a specific domain in the CUDA device context.

```rust
pub fn release_domain<F>(ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<F>
```

#### Parameters

- **`ctx`**: A reference to a `DeviceContext` specifying which device and stream the computation should be executed on.

#### Returns

The function returns an `IcicleResult<()>`, which represents the result of the operation. If the operation is successful, the function returns `Ok(())`, otherwise it returns an error.
