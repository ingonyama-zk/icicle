use crate::field::{ExtensionField, ScalarField};
use icicle_core::error::IcicleResult;
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use icicle_cuda_runtime::device::check_device;

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
      assert_eq!(device_id, ctx_device_id, "Device ids in domain_elements and context are different");
  }
  if let Some(device_id) = folded_eval.device_id() {
      assert_eq!(device_id, ctx_device_id, "Device ids in folded_eval and context are different");
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
      eval.len() as i32,
      &cfg as *const FriConfig,
    ).wrap()
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
      eval.len() as i32,
      &cfg as *const FriConfig,
    ).wrap()
  }
}

mod _fri {
  use super::{ExtensionField, ScalarField, CudaError, FriConfig};

  extern "C" {
      #[link_name = "m31_fold_line"]
      pub(crate) fn fold_line(
        line_eval: *const ExtensionField,
        domain_elements: *const ScalarField,
        alpha: &ExtensionField,
        folded_eval: *mut ExtensionField,
        n: i32,
        cfg: *const FriConfig
      ) -> CudaError;

      #[link_name = "m31_fold_circle_into_line"]
      pub(crate) fn fold_circle_into_line(
        circle_eval: *const ExtensionField,
        domain_elements: *const ScalarField,
        alpha: &ExtensionField,
        folded_line_eval: *mut ExtensionField,
        n: i32,
        cfg: *const FriConfig
      ) -> CudaError;
  }
}

#[cfg(test)]
pub(crate) mod tests {
  use super::*;
  use crate::field::{ExtensionField, ScalarField};
  use icicle_core::traits::FieldImpl;
  use icicle_cuda_runtime::memory::HostSlice;
  use std::iter::zip;

  #[test]
  fn test_fold_line() {
    // All values are taken from https://github.com/starkware-libs/stwo/blob/f976890/crates/prover/src/core/fri.rs#L1005-L1037
    const DEGREE: usize = 8;

    let evals_raw: [u32; DEGREE] = [1358331652, 807347720, 543926930, 1585623140, 1753377641, 616790922, 630401694, 1294134897];
    let evals_as_extension = evals_raw
      .into_iter()
      .map(
        |val: u32| ExtensionField::from_u32(val)
      )
      .collect::<Vec<ExtensionField>>();
    let eval = HostSlice::from_slice(evals_as_extension.as_slice());

    let domain_raw: [u32; DEGREE/2] = [1179735656, 1241207368, 1415090252, 2112881577];
    let domain_as_scalar = domain_raw
      .into_iter()
      .map(
        |val: u32| ScalarField::from_u32(val)
      )
      .collect::<Vec<ScalarField>>();
    let domain_elements = HostSlice::from_slice(domain_as_scalar.as_slice());
  
    let alpha = ExtensionField::from_u32(19283);

    let mut folded_eval_raw = vec![ExtensionField::zero(); DEGREE/2];
    let folded_eval = HostSlice::from_mut_slice(folded_eval_raw.as_mut_slice());

    let cfg = FriConfig::default();

    let res = fold_line(
      eval,
      domain_elements,
      folded_eval,
      alpha,
      &cfg,
    );

    assert!(res.is_ok());

    let expected_folded_evals_raw: [u32; DEGREE/2] = [547848116, 1352534073, 2053322292, 341725613];
    let expected_folded_evals_extension = expected_folded_evals_raw
      .into_iter()
      .map(
        |val: u32| ExtensionField::from_u32(val)
      )
      .collect::<Vec<ExtensionField>>();
    let expected_folded_evals = expected_folded_evals_extension.as_slice();

    for (i, (folded_eval_val, expected_folded_eval_val)) in zip(folded_eval.as_slice(), expected_folded_evals).enumerate() {
      assert_eq!(folded_eval_val, expected_folded_eval_val, "Mismatch of folded eval at {i}");
    }
  }
  
  #[test]
  fn test_fold_circle_to_line() {
    // const LOG_DEGREE: u32 = 4;
    // let circle_evaluation = polynomial_evaluation(LOG_DEGREE, LOG_BLOWUP_FACTOR);
    // let alpha = SecureField::one();
    // let folded_domain = LineDomain::new(circle_evaluation.domain.half_coset);

    // let mut folded_evaluation = LineEvaluation::new_zero(folded_domain);
    // fold_circle_into_line(&mut folded_evaluation, &circle_evaluation, alpha);

    // assert_eq!(
    //     log_degree_bound(folded_evaluation),
    //     LOG_DEGREE - CIRCLE_TO_LINE_FOLD_STEP
    // );
  }
}