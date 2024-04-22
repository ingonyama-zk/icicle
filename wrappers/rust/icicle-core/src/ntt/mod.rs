use icicle_cuda_runtime::device::check_device;
use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

pub use crate::{error::IcicleResult, traits::FieldImpl};

#[cfg(feature = "arkworks")]
#[doc(hidden)]
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
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NttAlgorithm {
    Auto,
    Radix2,
    MixedRadix,
}

/// Struct that encodes NTT parameters to be passed into the [ntt](ntt) function.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NTTConfig<'a, S> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](DeviceContext).
    pub ctx: DeviceContext<'a>,
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
    /// Explicitly select the NTT algorithm. Default value: Auto (the implementation selects radix-2 or mixed-radix algorithm based
    /// on heuristics).
    pub ntt_algorithm: NttAlgorithm,
}

impl<'a, S: FieldImpl> Default for NTTConfig<'a, S> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a, S: FieldImpl> NTTConfig<'a, S> {
    pub fn default_for_device(device_id: usize) -> Self {
        NTTConfig {
            ctx: DeviceContext::default_for_device(device_id),
            coset_gen: S::one(),
            batch_size: 1,
            columns_batch: false,
            ordering: Ordering::kNN,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
            ntt_algorithm: NttAlgorithm::Auto,
        }
    }
}

#[doc(hidden)]
pub trait NTTDomain<F: FieldImpl> {
    fn get_root_of_unity(max_size: u64) -> F;
    fn initialize_domain(primitive_root: F, ctx: &DeviceContext, fast_twiddles: bool) -> IcicleResult<()>;
    fn release_domain(ctx: &DeviceContext) -> IcicleResult<()>;
}

#[doc(hidden)]
pub trait NTT<T, F: FieldImpl>: NTTDomain<F> {
    fn ntt_unchecked(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<F>,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> IcicleResult<()>;
    fn ntt_inplace_unchecked(
        inout: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        dir: NTTDir,
        cfg: &NTTConfig<F>,
    ) -> IcicleResult<()>;
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
) -> IcicleResult<()>
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
    let ctx_device_id = cfg
        .ctx
        .device_id;
    if let Some(device_id) = input.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in input and context are different"
        );
    }
    if let Some(device_id) = output.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in output and context are different"
        );
    }
    check_device(ctx_device_id);
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
) -> IcicleResult<()>
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
/// * `ctx` - GPU index and stream to perform the computation.
pub fn initialize_domain<F>(primitive_root: F, ctx: &DeviceContext, fast_twiddles: bool) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    <<F as FieldImpl>::Config as NTTDomain<F>>::initialize_domain(primitive_root, ctx, fast_twiddles)
}

pub fn release_domain<F>(ctx: &DeviceContext) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    <<F as FieldImpl>::Config as NTTDomain<F>>::release_domain(ctx)
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
            #[link_name = concat!($field_prefix, concat!($ntt_type_lit, "Cuda"))]
            fn ntt_cuda(
                input: *const $inout,
                size: i32,
                dir: NTTDir,
                config: &NTTConfig<$domain_field>,
                output: *mut $inout,
            ) -> CudaError;
        }

        impl $ntt_type<$inout, $domain_field> for $domain_config {
            fn ntt_unchecked(
                input: &(impl HostOrDeviceSlice<$inout> + ?Sized),
                dir: NTTDir,
                cfg: &NTTConfig<$domain_field>,
                output: &mut (impl HostOrDeviceSlice<$inout> + ?Sized),
            ) -> IcicleResult<()> {
                unsafe {
                    ntt_cuda(
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
            ) -> IcicleResult<()> {
                unsafe {
                    ntt_cuda(
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
                #[link_name = concat!($field_prefix, "InitializeDomain")]
                fn initialize_ntt_domain(
                    primitive_root: &$field,
                    ctx: &DeviceContext,
                    fast_twiddles_mode: bool,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "ReleaseDomain")]
                fn release_ntt_domain(ctx: &DeviceContext) -> CudaError;

                #[link_name = concat!($field_prefix, "GetRootOfUnity")]
                fn get_root_of_unity(max_size: u64) -> $field;
            }

            impl NTTDomain<$field> for $field_config {
                fn initialize_domain(
                    primitive_root: $field,
                    ctx: &DeviceContext,
                    fast_twiddles: bool,
                ) -> IcicleResult<()> {
                    unsafe { initialize_ntt_domain(&primitive_root, ctx, fast_twiddles).wrap() }
                }

                fn release_domain(ctx: &DeviceContext) -> IcicleResult<()> {
                    unsafe { release_ntt_domain(ctx).wrap() }
                }

                fn get_root_of_unity(max_size: u64) -> $field {
                    unsafe { get_root_of_unity(max_size) }
                }
            }

            impl_ntt_without_domain!($field_prefix, $field, $field_config, NTT, "NTT", $field);
        }
    };
}

#[macro_export]
macro_rules! impl_ntt_tests {
    (
      $field:ident
    ) => {
        const MAX_SIZE: u64 = 1 << 17;
        static INIT: OnceLock<()> = OnceLock::new();
        static RELEASE: OnceLock<()> = OnceLock::new(); // for release domain test
        const FAST_TWIDDLES_MODE: bool = false;

        #[test]
        #[parallel]
        fn test_ntt() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_ntt::<$field>()
        }

        #[test]
        #[parallel]
        fn test_ntt_coset_from_subgroup() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_ntt_coset_from_subgroup::<$field>()
        }

        #[test]
        #[parallel]
        fn test_ntt_arbitrary_coset() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_ntt_arbitrary_coset::<$field>()
        }

        #[test]
        #[parallel]
        fn test_ntt_batch() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_ntt_batch::<$field>()
        }

        #[test]
        #[parallel]
        fn test_ntt_device_async() {
            // init_domain is in this test is performed per-device
            check_ntt_device_async::<$field>()
        }

        #[test]
        #[serial]
        fn test_ntt_release_domain() {
            INIT.get_or_init(move || init_domain::<$field>(MAX_SIZE, DEFAULT_DEVICE_ID, FAST_TWIDDLES_MODE));
            check_release_domain::<$field>()
        }
    };
}
