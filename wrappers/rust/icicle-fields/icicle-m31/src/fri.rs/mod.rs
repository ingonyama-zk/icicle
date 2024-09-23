use crate::field::{ExtensionField, ScalarField};
use icicle_core::error::IcicleResult;
use icicle_cuda_runtime::device_context::{DeviceContext, DEFAULT_DEVICE_ID};
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

/// Struct that encodes VecOps parameters.
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
) -> VecOpsConfig<'a> {
  if eval.len() != domain_elements.len() / 2 {
      panic!(
        "Number of domain elements is not half of the evaluation's domain size; {} != {} / 2",
        eval.len(),
        domain_elements.len()
      );
  }
  
  if eval.len() != folded_eval.len() / 2 {
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

pub fn fold_line<E: ExtensionField, S: ScalarField>(
  eval: &(impl HostOrDeviceSlice<E> + ?Sized),
  domain_elements: &(impl HostOrDeviceSlice<S> + ?Sized),
  folded_eval: &mut (impl HostOrDeviceSlice<E> + ?Sized),
  alpha: ExtensionField,
  cfg: &FriConfig<'a>,
) -> IcicleResult<()> {
  let cfg = check_fri_args(eval, domain_elements, folded_eval, cfg);
  _fri::fold_line(
    eval.as_ptr(),
    domain_elements.as_ptr(),
    &alpha,
    folded_eval.as_mut_ptr(),
    eval.len(),
    cfg,
  )
}

pub fn fold_circle_into_line<E: ExtensionField, S: ScalarField>(
  eval: &(impl HostOrDeviceSlice<E> + ?Sized),
  domain_elements: &(impl HostOrDeviceSlice<S> + ?Sized),
  folded_eval: &mut (impl HostOrDeviceSlice<E> + ?Sized),
  alpha: ExtensionField,
  cfg: &FriConfig<'a>,
) -> IcicleResult<()> {
  let cfg = check_fri_args(eval, domain_elements, folded_eval, cfg);
  _fri::fold_circle_into_line(
    eval.as_ptr(),
    domain_elements.as_ptr(),
    &alpha,
    folded_eval.as_mut_ptr(),
    eval.len(),
    cfg,
  )
}

mod _fri {
  use super::{$field, CudaError, Field, DeviceContext, FriConfig};

  extern "C" {
      #[link_name = concat!($curve_prefix, "_fold_line")]
      pub(crate) fn fold_line(
        line_eval: *const ExtensionField,
        domain_elements: *const ScalarField,
        alpha: const ExtensionField,
        folded_eval: *mut ExtensionField,
        n: int,
        cfg: *const FriConfig
      ) -> CudaError;

      #[link_name = concat!($curve_prefix, "_fold_circle_into_line")]
      pub(crate) fn fold_circle_into_line(
        circle_eval: *const ExtensionField,
        domain_elements: *const ScalarField,
        alpha: const ExtensionField,
        folded_line_eval: *mut ExtensionField,
        n: int,
        cfg: *const FriConfig
      ) -> CudaError;
  }
}

#[cfg(test)]
pub(crate) mod tests {
  use crate::field::{ExtensionField, ScalarField};
}