use crate::field::{ExtensionField, ScalarField};
use icicle_core::error::IcicleResult;
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device::check_device;
use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

/// Struct that encodes FRI parameters.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct FriConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,
    are_evals_on_device: bool,
    are_domain_elements_on_device: bool,
    are_results_on_device: bool,
    /// Whether to run the vector operations asynchronously. If set to `true`, the functions will be non-blocking and you'd need to synchronize
    /// it explicitly by running `stream.synchronize()`. If set to false, the functions will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for FriConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> FriConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        FriConfig {
            ctx: DeviceContext::default_for_device(device_id),
            are_evals_on_device: false,
            are_domain_elements_on_device: false,
            are_results_on_device: false,
            is_async: false,
        }
    }
}

fn check_fri_args<'a, F, S>(
    eval: &(impl HostOrDeviceSlice<F> + ?Sized),
    domain_elements: &(impl HostOrDeviceSlice<S> + ?Sized),
    folded_eval: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &FriConfig<'a>,
) -> FriConfig<'a> {
    if eval.len() / 2 != domain_elements.len() {
        panic!(
            "Number of domain elements is not half of the evaluation's domain size; {} != {} / 2",
            eval.len(),
            domain_elements.len()
        );
    }

    if eval.len() / 2 != folded_eval.len() {
        panic!(
            "Folded poly degree is not half of the evaluation poly's degree; {} != {} / 2",
            eval.len(),
            folded_eval.len()
        );
    }

    let ctx_device_id = cfg
        .ctx
        .device_id;

    if let Some(device_id) = eval.device_id() {
        assert_eq!(device_id, ctx_device_id, "Device ids in eval and context are different");
    }
    if let Some(device_id) = domain_elements.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in domain_elements and context are different"
        );
    }
    if let Some(device_id) = folded_eval.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in folded_eval and context are different"
        );
    }
    check_device(ctx_device_id);

    let mut res_cfg = cfg.clone();
    res_cfg.are_evals_on_device = eval.is_on_device();
    res_cfg.are_domain_elements_on_device = domain_elements.is_on_device();
    res_cfg.are_results_on_device = folded_eval.is_on_device();
    res_cfg
}

pub fn fold_line(
    eval: &(impl HostOrDeviceSlice<ExtensionField> + ?Sized),
    domain_elements: &(impl HostOrDeviceSlice<ScalarField> + ?Sized),
    folded_eval: &mut (impl HostOrDeviceSlice<ExtensionField> + ?Sized),
    alpha: ExtensionField,
    cfg: &FriConfig,
) -> IcicleResult<()> {
    let cfg = check_fri_args(eval, domain_elements, folded_eval, cfg);
    unsafe {
        _fri::fold_line(
            eval.as_ptr(),
            domain_elements.as_ptr(),
            &alpha,
            folded_eval.as_mut_ptr(),
            eval.len() as u64,
            &cfg as *const FriConfig,
        )
        .wrap()
    }
}

pub fn fold_circle_into_line(
    eval: &(impl HostOrDeviceSlice<ExtensionField> + ?Sized),
    domain_elements: &(impl HostOrDeviceSlice<ScalarField> + ?Sized),
    folded_eval: &mut (impl HostOrDeviceSlice<ExtensionField> + ?Sized),
    alpha: ExtensionField,
    cfg: &FriConfig,
) -> IcicleResult<()> {
    let cfg = check_fri_args(eval, domain_elements, folded_eval, cfg);
    unsafe {
        _fri::fold_circle_into_line(
            eval.as_ptr(),
            domain_elements.as_ptr(),
            &alpha,
            folded_eval.as_mut_ptr(),
            eval.len() as u64,
            &cfg as *const FriConfig,
        )
        .wrap()
    }
}

mod _fri {
    use super::{CudaError, ExtensionField, FriConfig, ScalarField};

    extern "C" {
        #[link_name = "m31_fold_line"]
        pub(crate) fn fold_line(
            line_eval: *const ExtensionField,
            domain_elements: *const ScalarField,
            alpha: &ExtensionField,
            folded_eval: *mut ExtensionField,
            n: u64,
            cfg: *const FriConfig,
        ) -> CudaError;

        #[link_name = "m31_fold_circle_into_line"]
        pub(crate) fn fold_circle_into_line(
            circle_eval: *const ExtensionField,
            domain_elements: *const ScalarField,
            alpha: &ExtensionField,
            folded_line_eval: *mut ExtensionField,
            n: u64,
            cfg: *const FriConfig,
        ) -> CudaError;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::traits::FieldImpl;
    use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
    use std::iter::zip;

    #[test]
    fn test_fold_line() {
        // All hardcoded values were generated with https://github.com/starkware-libs/stwo/blob/f976890/crates/prover/src/core/fri.rs#L1005-L1037
        const DEGREE: usize = 8;

        // Set evals
        let evals_raw: [u32; DEGREE] = [
            1358331652, 807347720, 543926930, 1585623140, 1753377641, 616790922, 630401694, 1294134897,
        ];
        let evals_as_extension = evals_raw
            .into_iter()
            .map(|val: u32| ExtensionField::from_u32(val))
            .collect::<Vec<ExtensionField>>();
        let eval = HostSlice::from_slice(evals_as_extension.as_slice());
        let mut d_eval = DeviceVec::<ExtensionField>::cuda_malloc(DEGREE).unwrap();
        d_eval
            .copy_from_host(eval)
            .unwrap();

        // Set domain
        let domain_raw: [u32; DEGREE / 2] = [1179735656, 1241207368, 1415090252, 2112881577];
        let domain_as_scalar = domain_raw
            .into_iter()
            .map(|val: u32| ScalarField::from_u32(val))
            .collect::<Vec<ScalarField>>();
        let domain_elements = HostSlice::from_slice(domain_as_scalar.as_slice());
        let mut d_domain_elements = DeviceVec::<ScalarField>::cuda_malloc(DEGREE / 2).unwrap();
        d_domain_elements
            .copy_from_host(domain_elements)
            .unwrap();

        // Alloc folded_evals
        let mut folded_eval_raw = vec![ExtensionField::zero(); DEGREE / 2];
        let folded_eval = HostSlice::from_mut_slice(folded_eval_raw.as_mut_slice());
        let mut d_folded_eval = DeviceVec::<ExtensionField>::cuda_malloc(DEGREE / 2).unwrap();

        let alpha = ExtensionField::from_u32(19283);
        let cfg = FriConfig::default();

        let res = fold_line(&d_eval[..], &d_domain_elements[..], &mut d_folded_eval[..], alpha, &cfg);

        assert!(res.is_ok());

        let expected_folded_evals_raw: [u32; DEGREE / 2] = [547848116, 1352534073, 2053322292, 341725613];
        let expected_folded_evals_extension = expected_folded_evals_raw
            .into_iter()
            .map(|val: u32| ExtensionField::from_u32(val))
            .collect::<Vec<ExtensionField>>();
        let expected_folded_evals = expected_folded_evals_extension.as_slice();

        d_folded_eval
            .copy_to_host(folded_eval)
            .unwrap();

        for (i, (folded_eval_val, expected_folded_eval_val)) in
            zip(folded_eval.as_slice(), expected_folded_evals).enumerate()
        {
            assert_eq!(
                folded_eval_val, expected_folded_eval_val,
                "Mismatch of folded eval at {i}"
            );
        }
    }

    #[test]
    fn test_fold_circle_to_line() {
        // All hardcoded values were generated with https://github.com/starkware-libs/stwo/blob/f976890/crates/prover/src/core/fri.rs#L1040-L1053
        const DEGREE: usize = 64;
        let circle_eval_raw: [u32; DEGREE] = [
            466407290, 127986842, 1870304883, 875137047, 1381744584, 1242514872, 1657247602, 1816542136, 18610701,
            183082621, 1291388290, 1665658712, 1768829380, 872721779, 1113994239, 827698214, 57598558, 1809783851,
            1582268514, 1018797774, 1927599636, 619773471, 802072749, 2111764399, 714973298, 532899888, 671071637,
            536208302, 1268828963, 255940280, 586928868, 535875357, 1650651309, 1473550629, 1387441966, 893930940,
            126593346, 1263510627, 18204497, 211871416, 604224095, 465540164, 1007455733, 755529771, 2130798047,
            871433949, 1073797249, 1097851807, 369407795, 302384846, 1904956607, 1168797665, 352925744, 10934213,
            409562797, 1646664722, 676414749, 35135895, 2606032, 2121020146, 1205801045, 1079025338, 2111544534,
            1635203417,
        ];
        let circle_eval_as_extension = circle_eval_raw
            .into_iter()
            .map(|val: u32| ExtensionField::from_u32(val))
            .collect::<Vec<ExtensionField>>();
        let circle_eval = HostSlice::from_slice(circle_eval_as_extension.as_slice());
        let mut d_circle_eval = DeviceVec::<ExtensionField>::cuda_malloc(DEGREE).unwrap();
        d_circle_eval
            .copy_from_host(circle_eval)
            .unwrap();

        let domain_raw: [u32; DEGREE / 2] = [
            1774253895, 373229752, 1309288441, 838195206, 262191051, 1885292596, 408478793, 1739004854, 212443077,
            1935040570, 1941424532, 206059115, 883753057, 1263730590, 350742286, 1796741361, 404685994, 1742797653,
            7144319, 2140339328, 68458636, 2079025011, 2137679949, 9803698, 228509164, 1918974483, 2132953617,
            14530030, 134155457, 2013328190, 1108537731, 1038945916,
        ];
        let domain_as_scalar = domain_raw
            .into_iter()
            .map(|val: u32| ScalarField::from_u32(val))
            .collect::<Vec<ScalarField>>();
        let domain_elements = HostSlice::from_slice(domain_as_scalar.as_slice());
        let mut d_domain_elements = DeviceVec::<ScalarField>::cuda_malloc(DEGREE / 2).unwrap();
        d_domain_elements
            .copy_from_host(domain_elements)
            .unwrap();

        let mut folded_eval_raw = vec![ExtensionField::zero(); DEGREE / 2];
        let folded_eval = HostSlice::from_mut_slice(folded_eval_raw.as_mut_slice());
        let mut d_folded_eval = DeviceVec::<ExtensionField>::cuda_malloc(DEGREE / 2).unwrap();

        let alpha = ExtensionField::one();
        let cfg = FriConfig::default();

        let res = fold_circle_into_line(
            &d_circle_eval[..],
            &d_domain_elements[..],
            &mut d_folded_eval[..],
            alpha,
            &cfg,
        );

        assert!(res.is_ok());

        let expected_folded_evals_raw: [u32; DEGREE / 2] = [
            1188788264, 1195916566, 953551618, 505128535, 403386644, 1619126710, 988135024, 1735901259, 1587281171,
            907165282, 799778920, 1532707002, 348262725, 267076231, 902054839, 98124803, 1953436582, 267778518,
            632724299, 460151826, 2139528518, 1378487361, 1709496698, 48330818, 1343585282, 1852541250, 727719914,
            1964971391, 1423101288, 2099768709, 274685472, 1051044961,
        ];
        let expected_folded_evals_extension = expected_folded_evals_raw
            .into_iter()
            .map(|val: u32| ExtensionField::from_u32(val))
            .collect::<Vec<ExtensionField>>();
        let expected_folded_evals = expected_folded_evals_extension.as_slice();

        d_folded_eval
            .copy_to_host(folded_eval)
            .unwrap();

        for (i, (folded_eval_val, expected_folded_eval_val)) in
            zip(folded_eval.as_slice(), expected_folded_evals).enumerate()
        {
            assert_eq!(
                folded_eval_val, expected_folded_eval_val,
                "Mismatch of folded eval at {i}"
            );
        }
    }
}
