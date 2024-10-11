use icicle_core::error::IcicleResult;
use icicle_core::ntt::{NTTConfig, NTTDir, Ordering};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device::check_device;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::field::{ComplexExtensionField, ScalarField};
use icicle_core::ntt::FieldImpl;

extern "C" {
    #[link_name = "m31_initialize_domain"]
    fn initialize_domain(primitive_root: &ComplexExtensionField, ctx: &DeviceContext) -> CudaError;

    #[link_name = "m31_release_domain"]
    fn release_ntt_domain(ctx: &DeviceContext) -> CudaError;

    #[link_name = "m31_get_root_of_unity"]
    fn get_root_of_unity(max_size: u32, rou_out: *mut ComplexExtensionField);

    #[link_name = "m31_ntt_cuda"]
    fn ntt_cuda(
        input: *const ScalarField,
        size: i32,
        dir: NTTDir,
        config: &NTTConfig<ScalarField>,
        output: *mut ScalarField,
    ) -> CudaError;
}

/// Generates twiddle factors which will be used to compute NTTs.
///
/// # Arguments
///
/// * `primitive_root` - primitive root to generate twiddles from. Should be of large enough order to cover all
/// DCCTs that you need. For example, if DCCTs of sizes 2^17 and 2^18 are computed, use the primitive root of order 2^18.
/// This function will panic if the order of `primitive_root` is not a power of two.
///
/// * `ctx` - GPU index and stream to perform the computation.
pub fn initialize_dcct_domain(primitive_root: ComplexExtensionField, ctx: &DeviceContext) -> IcicleResult<()> {
    unsafe { initialize_domain(&primitive_root, ctx).wrap() }
}

pub fn release_domain(ctx: &DeviceContext) -> IcicleResult<()> {
    unsafe { release_ntt_domain(ctx).wrap() }
}

pub fn get_dcct_root_of_unity(max_size: u32) -> ComplexExtensionField {
    let mut rou = ComplexExtensionField::zero();
    unsafe { get_root_of_unity(max_size, &mut rou as *mut ComplexExtensionField) };
    rou
}

fn dcct_unchecked(
    input: &(impl HostOrDeviceSlice<ScalarField> + ?Sized),
    dir: NTTDir,
    cfg: &NTTConfig<ScalarField>,
    output: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
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

/// Computes the DCCT, or a batch of several DCCTs.
///
/// # Arguments
///
/// * `input` - inputs of the DCCT.
///
/// * `dir` - whether to compute forward of inverse DCCT.
///
/// * `cfg` - config used to specify extra arguments of the DCCT.
///
/// * `output` - buffer to write the DCCT outputs into. Must be of the same size as `input`.
pub fn dcct(
    input: &(impl HostOrDeviceSlice<ScalarField> + ?Sized),
    dir: NTTDir,
    cfg: &NTTConfig<ScalarField>,
    output: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
) -> IcicleResult<()> {
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

    dcct_unchecked(input, dir, &local_cfg, output)
}

pub fn evaluate(
    input: &(impl HostOrDeviceSlice<ScalarField> + ?Sized),
    cfg: &NTTConfig<ScalarField>,
    output: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
) -> IcicleResult<()> {
    let mut local_cfg = cfg.clone();
    local_cfg.ordering = Ordering::kNM;
    dcct(input, NTTDir::kForward, &local_cfg, output)
}

pub fn interpolate(
    input: &(impl HostOrDeviceSlice<ScalarField> + ?Sized),
    cfg: &NTTConfig<ScalarField>,
    output: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
) -> IcicleResult<()> {
    let mut local_cfg = cfg.clone();
    local_cfg.ordering = Ordering::kMN;
    dcct(input, NTTDir::kInverse, &local_cfg, output)
}

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::ntt::{FieldImpl, NTTConfig};
    use icicle_cuda_runtime::{device_context::DeviceContext, memory::HostSlice};

    use crate::{
        dcct::{evaluate, get_dcct_root_of_unity, initialize_dcct_domain},
        field::ScalarField,
    };

    #[test]
    #[ignore = "cuda mem err"]
    fn test_evaluate_4() {
        const LOG: u32 = 4;

        let rou = get_dcct_root_of_unity(LOG);
        println!("ROU {:?}", rou);
        initialize_dcct_domain(rou, &DeviceContext::default()).unwrap();
        println!("initialied DCCT succesfully");

        let coeffs: Vec<ScalarField> = (0u32..1 << LOG)
            .map(ScalarField::from_u32)
            .collect();
        let expected = [
            0x6103dfb, 0x25cbfdad, 0x4f35cf25, 0x39a34de3, 0x24862591, 0x7607aa42, 0xc8f98cb, 0x23c949af, 0x33a8d92e,
            0x687b1fd, 0x56dd7a35, 0x1a447068, 0x4926b47e, 0x4c71dfd9, 0x22a218e1, 0x2485ad20, 0x284f0ee4, 0x3dfabc33,
            0x69d57fdf, 0x65aad3ef, 0x3a80d4a1, 0x5157f85f, 0x6c3182de, 0x6294b4ff, 0x13e77c3a, 0x1ce9cc10, 0x7ae749ca,
            0x631b8c6c, 0x5bfa6c0b, 0x670d13c7, 0x20a57b4a, 0x7566e736, 0x23362583, 0x4e85b831, 0x4be15511, 0x5154ecb6,
            0x3890cf71, 0x74a836f4, 0x73738fc3, 0x3f454b45, 0x612d2b5f, 0x61c716ce, 0x53d3e3fe, 0x961aae5, 0x329b7b98,
            0x5531aafc, 0x4dfbdb78, 0x1b6dcced, 0x6535b85a, 0x5e402279, 0x533c9f36, 0x737b7774, 0x71e33235, 0x681fa712,
            0x399d1b64, 0x3de2ed65, 0x536c73a8, 0x10d8a6cc, 0x2ac55f9b, 0x558ea2a6, 0x76a5ce55, 0x4f7d8990, 0x1790f920,
            0x61bd1df4,
        ]
        .map(ScalarField::from_u32);

        let mut evaluations = vec![ScalarField::zero(); 1 << LOG];
        let cfg = NTTConfig::default();
        evaluate(
            HostSlice::from_slice(&coeffs),
            &cfg,
            HostSlice::from_mut_slice(&mut evaluations),
        )
        .unwrap();

        assert_eq!(evaluations, expected);
    }
}
