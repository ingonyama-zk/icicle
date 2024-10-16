use icicle_core::error::IcicleResult;
use icicle_core::ntt::{NTTConfig, NTTDir};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device::check_device;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::field::{ComplexExtensionField, ScalarField};
use icicle_core::ntt::FieldImpl;

extern "C" {
    #[link_name = "m31_initialize_domain"]
    fn initialize_domain(logn: u32, primitive_root: &ComplexExtensionField, ctx: &DeviceContext) -> CudaError;

    #[link_name = "m31_release_domain"]
    fn release_ntt_domain(logn: u32, ctx: &DeviceContext) -> CudaError;

    #[link_name = "m31_get_root_of_unity"]
    fn get_root_of_unity(max_size: u64, rou_out: *mut ComplexExtensionField);

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
pub fn initialize_dcct_domain(
    logn: u32,
    primitive_root: ComplexExtensionField,
    ctx: &DeviceContext,
) -> IcicleResult<()> {
    unsafe { initialize_domain(logn, &primitive_root, ctx).wrap() }
}

pub fn release_domain(logn: u32, ctx: &DeviceContext) -> IcicleResult<()> {
    unsafe { release_ntt_domain(logn, ctx).wrap() }
}

pub fn get_dcct_root_of_unity(max_size: u64) -> ComplexExtensionField {
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
    dcct(input, NTTDir::kForward, &cfg, output)
}

pub fn interpolate(
    input: &(impl HostOrDeviceSlice<ScalarField> + ?Sized),
    cfg: &NTTConfig<ScalarField>,
    output: &mut (impl HostOrDeviceSlice<ScalarField> + ?Sized),
) -> IcicleResult<()> {
    dcct(input, NTTDir::kInverse, &cfg, output)
}

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::ntt::{FieldImpl, NTTConfig, Ordering};
    use icicle_cuda_runtime::{device_context::DeviceContext, memory::HostSlice};

    use crate::{
        dcct::{evaluate, get_dcct_root_of_unity, initialize_dcct_domain, interpolate},
        field::ScalarField,
    };

    #[test]
    fn test_evaluate_4() {
        const LOG: u32 = 4;

        let rou = get_dcct_root_of_unity(1 << LOG);
        println!("ROU {:?}", rou);
        initialize_dcct_domain(LOG, rou, &DeviceContext::default()).unwrap();
        println!("initialied DCCT succesfully");

        let coeffs: Vec<ScalarField> = (0u32..1 << LOG)
            .map(ScalarField::from_u32)
            .collect();

        let expected = [
            0x4fe32dfe,
            0x38de5f54,
            0x71b8fbe6,
            0x7e54ce84,
            0x40f40f27,
            0x7374451f,
            0x3d7c46eb,
            0x154c0cff,
            0x234c23f4,
            0x698b7dae,
            0x50cf78c7,
            0x7db21e80,
            0x5489c73d,
            0x5451156d,
            0x776e1c0d,
            0x45dce6a
        ]
        .map(ScalarField::from_u32);

        let mut evaluations = vec![ScalarField::zero(); 1 << LOG];
        let mut cfg = NTTConfig::default();
        cfg.ordering = Ordering::kNR;
        evaluate(
            HostSlice::from_slice(&coeffs),
            &cfg,
            HostSlice::from_mut_slice(&mut evaluations),
        )
        .unwrap();

        assert_eq!(evaluations, expected);
    }

    #[test]
    fn test_interpolate_4() {
        const LOG: u32 = 4;

        let rou = get_dcct_root_of_unity(1 << LOG);
        println!("ROU {:?}", rou);
        initialize_dcct_domain(LOG, rou, &DeviceContext::default()).unwrap();
        println!("initialied DCCT succesfully");

        let evaluations: Vec<ScalarField> = (0u32..1 << LOG)
            .map(ScalarField::from_u32)
            .collect();

        let expected = [
            0x40000007,
            0x000000,
            0x73d34ffa,
            0x000000,
            0x16af762a,
            0x000000,
            0x882b425,
            0x000000,
            0x7ffbffff,
            0x000000,
            0x41ff8ac,
            0x000000,
            0x5c76e64c,
            0x000000,
            0x5a82de7a,
            0x000004,
        ]
        .map(ScalarField::from_u32);

        let mut coeffs = vec![ScalarField::zero(); 1 << LOG];
        let cfg = NTTConfig::default();
        interpolate(
            HostSlice::from_slice(&evaluations),
            &cfg,
            HostSlice::from_mut_slice(&mut coeffs),
        )
        .unwrap();

        assert_eq!(coeffs, expected);
    }
}
