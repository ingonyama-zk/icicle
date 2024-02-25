# Supporting Additional Curves

We understand the need for ZK developers to use different curves, some common some more exotic. For this reason we designed ICICLE to allow developers to add any curve they desire.

## ICICLE Core

ICICLE core is very generic by design so all algorithms and primitives are designed to work based of configuration files [selected during compile](https://github.com/ingonyama-zk/icicle/blob/main/icicle/curves/curve_config.cuh) time. This is why we compile ICICLE Core per curve.

To add support a new curve you must create a new file under [`icicle/curves`](https://github.com/ingonyama-zk/icicle/tree/main/icicle/curves). The file should be named `<curve_name>_params.cuh`.

We also require some changes to [`curve_config.cuh`](https://github.com/ingonyama-zk/icicle/blob/main/icicle/curves/curve_config.cuh#L16-L29), we need to add a new curve id.

```
...

#define BN254     1
#define BLS12_381 2
#define BLS12_377 3
#define BW6_761   4
#define GRUMPKIN  5
#define <curve_name> 6

...
```

Make sure to modify the [rest of the file](https://github.com/ingonyama-zk/icicle/blob/4beda3a900eda961f39af3a496f8184c52bf3b41/icicle/curves/curve_config.cuh#L16-L29) accordingly.

Finally we must modify the [`make` file](https://github.com/ingonyama-zk/icicle/blob/main/icicle/CMakeLists.txt#L64) to make sure we can compile our new curve.

```
set(SUPPORTED_CURVES bn254;bls12_381;bls12_377;bw6_761;<curve_name>)
```

## Bindings

In order to support a new curves in the binding libraries you first must support it in ICICLE core.

### Rust

Create a new folder named `icicle-<curve_name>` under the [rust wrappers folder](https://github.com/ingonyama-zk/icicle/tree/main/wrappers/rust/icicle-curves). Your new directory should look like this.

```
└── rust
    ├── icicle-curves
        ├── icicle-<curve_name>
    │   │   ├── Cargo.toml
    │   │   ├── build.rs
    │   │   └── src/
    │   │       ├── curve.rs
    │   │       ├── lib.rs
    │   │       ├── msm/
    │   │       │   └── mod.rs
    │   │       └── ntt/
    │   │           └── mod.rs
```

Lets look at [`ntt/mod.rs`](https://github.com/ingonyama-zk/icicle/blob/main/wrappers/rust/icicle-curves/icicle-bn254/src/ntt/mod.rs) for example.

```
...

extern "C" {
    #[link_name = "bn254NTTCuda"]
    fn ntt_cuda<'a>(
        input: *const ScalarField,
        size: usize,
        is_inverse: bool,
        config: &NTTConfig<'a, ScalarField>,
        output: *mut ScalarField,
    ) -> CudaError;

    #[link_name = "bn254DefaultNTTConfig"]
    fn default_ntt_config() -> NTTConfig<'static, ScalarField>;

    #[link_name = "bn254InitializeDomain"]
    fn initialize_ntt_domain(primitive_root: ScalarField, ctx: &DeviceContext) -> CudaError;
}

...
```

Here you would need to replace `bn254NTTCuda` with `<curve_name>NTTCuda`. Most of these changes are pretty straight forward. One thing you should pay attention to is limb sizes as these change for different curves. For example `BN254` [has limb size of 8](https://github.com/ingonyama-zk/icicle/blob/4beda3a900eda961f39af3a496f8184c52bf3b41/wrappers/rust/icicle-curves/icicle-bn254/src/curve.rs#L15) but for your curve this may be different.

### Golang

Golang is WIP in v1, coming soon. Please checkout a previous [release v0.1.0](https://github.com/ingonyama-zk/icicle/releases/tag/v0.1.0) for golang bindings.
