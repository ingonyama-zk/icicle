use crate::traits::FieldImpl;
use icicle_runtime::{
    config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStreamHandle,
};

pub mod tests;

/// Whether to perform normal forward NTT, or inverse NTT (iNTT). Mathematically, forward NTT computes polynomial
/// evaluations from coefficients while inverse NTT computes coefficients from evaluations.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NTTDir {
    kForward,
    kInverse,
}

/// How to order inputs and outputs of the NTT. If needed, use this field to specify decimation: decimation in time
/// (DIT) corresponds to `Ordering::kRN` while decimation in frequency (DIF) to `Ordering::kNR`. Also, to specify
/// butterfly to be used, select `Ordering::kRN` for Cooley-Tukey and `Ordering::kNR` for Gentleman-Sande. There's
/// no implication that a certain decimation or butterfly will actually be used under the hood, this is just for
/// compatibility with codebases that use "decimation" and "butterfly" to denote ordering of inputs and outputs.
///
/// Ordering options are:
/// - kNN: inputs and outputs are natural-order (example of natural ordering: `a_0, a_1, a_2, a_3, a_4, a_5, a_6,
/// a_7`.
/// - kNR: inputs are natural-order and outputs are bit-reversed-order (example of bit-reversed ordering: `a_0,
/// a_4, a_2, a_6, a_1, a_5, a_3, a_7`.
/// - kRN: inputs are bit-reversed-order and outputs are natural-order.
/// - kRR: inputs and outputs are bit-reversed-order.
///
/// Mixed-Radix NTT: digit-reversal is a generalization of bit-reversal where the latter is a special case with 1b
/// digits. Mixed-radix NTTs of different sizes would generate different reordering of inputs/outputs. Having said
/// that, for a given size N it is guaranteed that every two mixed-radix NTTs of size N would have the same
/// digit-reversal pattern. The following orderings kNM and kMN are conceptually like kNR and kRN but for
/// mixed-digit-reordering. Note that for the cases '(1) NTT, (2) elementwise ops and (3) INTT' kNM and kMN are most
/// efficient.
/// Note: kNR, kRN, kRR refer to the radix-2 NTT reversal pattern. Those cases are supported by mixed-radix NTT with
/// reduced efficiency compared to kNM and kMN.
/// - kNM: inputs are natural-order and outputs are digit-reversed-order (=mixed).
/// - kMN: inputs are digit-reversed-order (=mixed) and outputs are natural-order.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    kNN,
    kNR,
    kRN,
    kRR,
    kNM,
    kMN,
}

///Which NTT algorithm to use. options are:
///- Auto: implementation selects automatically based on heuristic. This value is a good default for most cases.
///- Radix2: explicitly select radix-2 NTT algorithm
///- MixedRadix: explicitly select mixed-radix NTT algorithm
///
// Note: CUDA specific config to be passed via config-extension
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NttAlgorithm {
    Auto,
    Radix2,
    MixedRadix,
}

// CUDA backend specific flags
pub const CUDA_NTT_FAST_TWIDDLES_MODE: &str = "fast_twiddles";
pub const CUDA_NTT_ALGORITHM: &str = "ntt_algorithm";

#[repr(C)]
#[derive(Debug, Clone)]
pub struct NTTConfig<S> {
    pub stream_handle: IcicleStreamHandle,
    /// Coset generator. Used to perform coset (i)NTTs. Default value: `S::one()` (corresponding to no coset being used).
    pub coset_gen: S,
    /// The number of NTTs to compute. Default value: 1.
    pub batch_size: i32,
    /// If true the function will compute the NTTs over the columns of the input matrix and not over the rows.
    pub columns_batch: bool,
    /// Ordering of inputs and outputs. See [Ordering](Ordering). Default value: `Ordering::kNN`.
    pub ordering: Ordering,
    pub are_inputs_on_device: bool,
    pub are_outputs_on_device: bool,
    /// Whether to run the NTT asynchronously. If set to `true`, the NTT function will be non-blocking and you'd need to synchronize
    /// it explicitly by running `stream.synchronize()`. If set to false, the NTT function will block the current CPU thread.
    pub is_async: bool,
    pub ext: ConfigExtension,
}

impl<S: FieldImpl> NTTConfig<S> {
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            coset_gen: S::one(),
            batch_size: 1,
            columns_batch: false,
            ordering: Ordering::kNN,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct NTTInitDomainConfig {
    pub stream_handle: IcicleStreamHandle,
    pub is_async: bool,
    pub ext: ConfigExtension,
}

impl NTTInitDomainConfig {
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

#[doc(hidden)]
pub trait NTTDomain<F: FieldImpl> {
    fn get_root_of_unity(max_size: u64) -> F;
    fn initialize_domain(primitive_root: F, config: &NTTInitDomainConfig) -> Result<(), eIcicleError>;
    fn release_domain() -> Result<(), eIcicleError>;
}

#[doc(hidden)]
pub trait NTT<T, F: FieldImpl>: NTTDomain<F> {
    fn ntt_unchecked(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<F>,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;
    fn ntt_inplace_unchecked(
        inout: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<F>,
    ) -> Result<(), eIcicleError>;
}

/// Computes the NTT, or a batch of several NTTs.
///
/// # Arguments
///
/// * `input` - inputs of the NTT.
///
/// * `dir` - whether to compute forward of inverse NTT.
///
/// * `cfg` - config used to specify extra arguments of the NTT.
///
/// * `output` - buffer to write the NTT outputs into. Must be of the same size as `input`.
pub fn ntt<T, F>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    dir: NTTDir,
    cfg: &NTTConfig<F>,
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<T, F>,
{
    if input.len() != output.len() {
        panic!(
            "input and output lengths {}; {} do not match",
            input.len(),
            output.len()
        );
    }

    // check device slices are on active device
    if input.is_on_device() && !input.is_on_active_device() {
        panic!("input not allocated on an inactive device");
    }
    if output.is_on_device() && !output.is_on_active_device() {
        panic!("output not allocated on an inactive device");
    }

    let mut local_cfg = cfg.clone();
    local_cfg.are_inputs_on_device = input.is_on_device();
    local_cfg.are_outputs_on_device = output.is_on_device();

    <<F as FieldImpl>::Config as NTT<T, F>>::ntt_unchecked(input, dir, &local_cfg, output)
}

/// Computes the NTT, or a batch of several NTTs inplace.
///
/// # Arguments
///
/// * `inout` - buffer with inputs to also write the NTT outputs into.
///
/// * `dir` - whether to compute forward of inverse NTT.
///
/// * `cfg` - config used to specify extra arguments of the NTT.
pub fn ntt_inplace<T, F>(
    inout: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    dir: NTTDir,
    cfg: &NTTConfig<F>,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTT<T, F>,
{
    let mut local_cfg = cfg.clone();
    local_cfg.are_inputs_on_device = inout.is_on_device();
    local_cfg.are_outputs_on_device = inout.is_on_device();

    <<F as FieldImpl>::Config as NTT<T, F>>::ntt_inplace_unchecked(inout, dir, &local_cfg)
}

/// Generates twiddle factors which will be used to compute NTTs.
///
/// # Arguments
///
/// * `primitive_root` - primitive root to generate twiddles from. Should be of large enough order to cover all
/// NTTs that you need. For example, if NTTs of sizes 2^17 and 2^18 are computed, use the primitive root of order 2^18.
/// This function will panic if the order of `primitive_root` is not a power of two.
///
pub fn initialize_domain<F>(primitive_root: F, config: &NTTInitDomainConfig) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    <<F as FieldImpl>::Config as NTTDomain<F>>::initialize_domain(primitive_root, config)
}

pub fn release_domain<F>() -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    <<F as FieldImpl>::Config as NTTDomain<F>>::release_domain()
}

pub fn get_root_of_unity<F>(max_size: u64) -> F
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    <<F as FieldImpl>::Config as NTTDomain<F>>::get_root_of_unity(max_size)
}

#[macro_export]
macro_rules! impl_ntt_without_domain {
    (
      $field_prefix:literal,
      $domain_field:ident,
      $domain_config:ident,
      $ntt_type:ident,
      $ntt_type_lit:literal,
      $inout:ident
    ) => {
        extern "C" {
            #[link_name = concat!($field_prefix, $ntt_type_lit)]
            fn ntt_ffi(
                input: *const $inout,
                size: i32,
                dir: NTTDir,
                config: &NTTConfig<$domain_field>,
                output: *mut $inout,
            ) -> eIcicleError;
        }

        impl $ntt_type<$inout, $domain_field> for $domain_config {
            fn ntt_unchecked(
                input: &(impl HostOrDeviceSlice<$inout> + ?Sized),
                dir: NTTDir,
                cfg: &NTTConfig<$domain_field>,
                output: &mut (impl HostOrDeviceSlice<$inout> + ?Sized),
            ) -> Result<(), eIcicleError> {
                unsafe {
                    ntt_ffi(
                        input.as_ptr(),
                        (input.len() / (cfg.batch_size as usize)) as i32,
                        dir,
                        cfg,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn ntt_inplace_unchecked(
                inout: &mut (impl HostOrDeviceSlice<$inout> + ?Sized),
                dir: NTTDir,
                cfg: &NTTConfig<$domain_field>,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    ntt_ffi(
                        inout.as_mut_ptr(),
                        (inout.len() / (cfg.batch_size as usize)) as i32,
                        dir,
                        cfg,
                        inout.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_ntt {
    (
      $field_prefix:literal,
      $field_prefix_ident:ident,
      $field:ident,
      $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use crate::ntt::*;

            extern "C" { 
                #[link_name = concat!($field_prefix, "_ntt_init_domain")]
                fn initialize_ntt_domain(primitive_root: &$field, config: &NTTInitDomainConfig) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_ntt_release_domain")]
                fn release_ntt_domain() -> eIcicleError;

                #[link_name = concat!($field_prefix, "_get_root_of_unity")]
                fn get_root_of_unity(max_size: u64, rou: *mut $field) -> eIcicleError;
            }

            impl NTTDomain<$field> for $field_config {
                fn initialize_domain(primitive_root: $field, config: &NTTInitDomainConfig) -> Result<(), eIcicleError> {
                    unsafe { initialize_ntt_domain(&primitive_root, config).wrap() }
                }

                fn release_domain() -> Result<(), eIcicleError> {
                    unsafe { release_ntt_domain().wrap() }
                }

                fn get_root_of_unity(max_size: u64) -> $field {
                    let mut rou = std::mem::MaybeUninit::<$field>::uninit(); // Prepare uninitialized memory for rou
                    unsafe {
                        get_root_of_unity(max_size, rou.as_mut_ptr());
                        return rou.assume_init();
                    }
                }
            }

            impl_ntt_without_domain!($field_prefix, $field, $field_config, NTT, "_ntt", $field);
        }
    };
}

#[macro_export]
macro_rules! impl_ntt_tests {
    (
      $field:ident
    ) => {
        use icicle_core::test_utilities;
        use icicle_runtime::{device::Device, runtime};
        use std::sync::Once;

        const MAX_SIZE: u64 = 1 << 18;
        static INIT: Once = Once::new();
        const FAST_TWIDDLES_MODE: bool = false;

        pub fn initialize() {
            INIT.call_once(move || {
                test_utilities::test_load_and_init_devices();
                // init domain for both devices
                test_utilities::test_set_ref_device();
                init_domain::<$field>(MAX_SIZE, FAST_TWIDDLES_MODE);

                test_utilities::test_set_main_device();
                init_domain::<$field>(MAX_SIZE, FAST_TWIDDLES_MODE);
            });
            test_utilities::test_set_main_device();
        }

        #[test]
        #[parallel]
        fn test_ntt() {
            initialize();
            check_ntt::<$field>()
        }

        #[test]
        #[parallel]
        fn test_ntt_coset_from_subgroup() {
            initialize();
            check_ntt_coset_from_subgroup::<$field>()
        }

        #[test]
        #[parallel]
        fn test_ntt_coset_interpolation_nm() {
            initialize();
            check_ntt_coset_interpolation_nm::<$field>();
        }

        #[test]
        #[parallel]
        fn test_ntt_arbitrary_coset() {
            initialize();
            check_ntt_arbitrary_coset::<$field>()
        }

        #[test]
        #[parallel]
        fn test_ntt_batch() {
            initialize();
            check_ntt_batch::<$field>()
        }

        #[test]
        #[parallel]
        fn test_ntt_device_async() {
            initialize();
            check_ntt_device_async::<$field>()
        }

        // problematic test since cannot have it execute last
        // also not testing much
        #[test]
        #[serial]
        fn test_ntt_release_domain() {
            // initialize();
            // check_release_domain::<$field>()
        }
    };
}

#[macro_export]
macro_rules! impl_ntt_bench {
    (
      $field_prefix:literal,
      $field:ident
    ) => {
        use std::{env, sync::OnceLock};
        use criterion::{black_box, criterion_group, criterion_main, Criterion};
        use icicle_runtime::{memory::{HostSlice,HostOrDeviceSlice},device::Device,is_device_available,  get_active_device, set_device, runtime::load_backend_from_env_or_default};
        use icicle_core::{
            ntt::{NTTConfig, NTTInitDomainConfig, NTTDir, NttAlgorithm, Ordering, NTTDomain, ntt, NTT},
            traits::{GenerateRandom,FieldImpl},
            vec_ops::VecOps,
        };

        static INIT: OnceLock<()> = OnceLock::new();

        fn load_and_init_backend_device() {
            // Attempt to load the backends
            let _ = load_backend_from_env_or_default(); // try loading from /opt/icicle/backend or env ${ICICLE_BACKEND_INSTALL_DIR}

            // Check if BENCH_TARGET is defined
            let target = env::var("BENCH_TARGET").unwrap_or_else(|_| {
                // If not defined, try CUDA first, fallback to CPU
                if is_device_available(&Device::new("CUDA", 0)) {
                    "CUDA".to_string()
                } else {
                    "CPU".to_string()
                }
            });

            // Initialize the device with the determined target
            let device = Device::new(&target, 0);
            set_device(&device).unwrap();

            println!("ICICLE benchmark with {:?}", device);
        }

        fn benchmark_ntt<T, F: FieldImpl>(c: &mut Criterion)
        where
        <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
        <F as FieldImpl>::Config: VecOps<F>,
        {
            use criterion::SamplingMode;
            use icicle_core::ntt::tests::init_domain;
            use std::env;

            load_and_init_backend_device();

            let group_id = format!("{} NTT", $field_prefix);
            let mut group = c.benchmark_group(&group_id);
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(10);

            const MAX_LOG2: u32 = 25; // max length = 2 ^ MAX_LOG2

            let max_log2 = env::var("MAX_LOG2")
                .unwrap_or_else(|_| MAX_LOG2.to_string())
                .parse::<u32>()
                .unwrap_or(MAX_LOG2);

            const FAST_TWIDDLES_MODE: bool = false;

            INIT.get_or_init(move || init_domain::<$field>(1 << max_log2, FAST_TWIDDLES_MODE));

            let coset_generators = [F::one(), F::Config::generate_random(1)[0]];
            let mut config = NTTConfig::<F>::default();

            for test_size_log2 in (13u32..=max_log2) {
                for batch_size_log2 in (7u32..17u32) {
                    let test_size = 1 << test_size_log2;
                    let batch_size = 1 << batch_size_log2;
                    let full_size = batch_size * test_size;

                    if full_size > 1 << max_log2 {
                        continue;
                    }

                    let scalars = F::Config::generate_random(full_size);
                    let input = HostSlice::from_slice(&scalars);

                    let mut batch_ntt_result = vec![F::zero(); batch_size * test_size];
                    let batch_ntt_result = HostSlice::from_mut_slice(&mut batch_ntt_result);
                    let mut config = NTTConfig::<F>::default();
                    for dir in [NTTDir::kForward, NTTDir::kInverse ] {
                        for ordering in [
                            Ordering::kNN,
                            Ordering::kNR,
                            Ordering::kRN,
                            Ordering::kRR,
                            Ordering::kNM,
                            Ordering::kMN,
                        ] {
                            config.ordering = ordering;
                            config.batch_size = batch_size as i32;
                            let bench_descr = format!(
                                "{:?} {:?} {} x {}",
                                ordering, dir, test_size, batch_size
                            );
                            group.bench_function(&bench_descr, |b| {
                                b.iter(|| {
                                    ntt::<F, F>(
                                        input,
                                        dir,
                                        &mut config,
                                        batch_ntt_result
                                    )
                                })
                            });
                        }
                    }
                }
            }

            group.finish();
        }

        criterion_group!(benches, benchmark_ntt<$field, $field>);
        criterion_main!(benches);
    };
}
