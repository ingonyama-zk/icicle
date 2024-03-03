# MSM

## How do I toggle between the supported algorithms?

When creating your MSM Config you may state which algorithm you wish to use. `is_big_triangle=true` will activate Large triangle accumulation and `is_big_triangle=false` will activate Bucket accumulation.

```rust
...

let mut cfg_bls12377 = msm::get_default_msm_config::<BLS12377CurveCfg>();

// is_big_triangle will determine which algorithm to use 
cfg_bls12377.is_big_triangle = true;

msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();
...
```

You may reference the rust code [here](https://github.com/ingonyama-zk/icicle/blob/77a7613aa21961030e4e12bf1c9a78a2dadb2518/wrappers/rust/icicle-core/src/msm/mod.rs#L54).


## How do I toggle between MSM modes?

Toggling between MSM modes occurs automatically based on the number of results you are expecting from the `msm::msm` function. If you are expecting an array of `msm_results`, ICICLE will automatically split `scalars` and `points` into equal parts and run them as multiple MSMs in parallel.

```rust
...

let mut msm_result: HostOrDeviceSlice<'_, G1Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
msm::msm(&scalars, &points, &cfg, &mut msm_result).unwrap();

...
```

In the example above we allocate a single expected result which the MSM method will interpret as `batch_size=1` and run a single MSM.


In the next example, we are expecting 10 results which sets `batch_size=10` and runs 10 MSMs in batch mode.

```rust
...

let mut msm_results: HostOrDeviceSlice<'_, G1Projective> = HostOrDeviceSlice::cuda_malloc(10).unwrap();
msm::msm(&scalars, &points, &cfg, &mut msm_results).unwrap();

...
```

Here is a [reference](https://github.com/ingonyama-zk/icicle/blob/77a7613aa21961030e4e12bf1c9a78a2dadb2518/wrappers/rust/icicle-core/src/msm/mod.rs#L108) to the code which automatically sets the batch size. For more MSM examples have a look [here](https://github.com/ingonyama-zk/icicle/blob/77a7613aa21961030e4e12bf1c9a78a2dadb2518/examples/rust/msm/src/main.rs#L1).

## Support for G2 group

MSM also supports G2 group. 

Using MSM in G2 requires a G2 config, and of course your Points should also be G2 Points.

```rust
... 

let scalars = HostOrDeviceSlice::Host(upper_scalars[..size].to_vec());
let g2_points = HostOrDeviceSlice::Host(g2_upper_points[..size].to_vec());
let mut g2_msm_results: HostOrDeviceSlice<'_, G2Projective> = HostOrDeviceSlice::cuda_malloc(1).unwrap();
let mut g2_cfg = msm::get_default_msm_config::<G2CurveCfg>();

msm::msm(&scalars, &g2_points, &g2_cfg, &mut g2_msm_results).unwrap();

...
```

Here you can [find an example](https://github.com/ingonyama-zk/icicle/blob/5a96f9937d0a7176d88c766bd3ef2062b0c26c37/examples/rust/msm/src/main.rs#L114) of MSM on G2 Points.
