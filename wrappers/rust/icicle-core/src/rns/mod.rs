// (1) define a trait for the RNS-conversions

use crate::{traits::FieldImpl, vec_ops::VecOpsConfig};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

pub trait RnsConversion<Zq: FieldImpl, ZqRns: FieldImpl> {
    fn to_rns(
        input: &(impl HostOrDeviceSlice<Zq> + ?Sized),
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<ZqRns> + ?Sized),
    ) -> Result<(), eIcicleError>;

    fn from_rns(
        input: &(impl HostOrDeviceSlice<ZqRns> + ?Sized),
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<Zq> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

// (2) implement generic RNS-conversions floating-functions that use the trait (<<F as FieldConfig> as RnsConversion>::to_rns(...), <<F as FieldConfig> as RnsConversion>::from_rns(...))
pub fn to_rns<Zq: FieldImpl, ZqRns: FieldImpl>(
    input: &(impl HostOrDeviceSlice<Zq> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<ZqRns> + ?Sized),
) -> Result<(), eIcicleError>
where
    <Zq as FieldImpl>::Config: RnsConversion<Zq, ZqRns>,
{
    if input.len() != output.len() {
        eprintln!(
            "input (size={}) and output (size={}) slices must have the same length",
            input.len(),
            output.len()
        );
        return Err(eIcicleError::InvalidArgument);
    }

    // check device slices are on active device
    if input.is_on_device() && !input.is_on_active_device() {
        eprintln!("input not allocated on an inactive device");
        return Err(eIcicleError::InvalidArgument);
    }
    if output.is_on_device() && !output.is_on_active_device() {
        eprintln!("output not allocated on an inactive device");
        return Err(eIcicleError::InvalidArgument);
    }

    let mut cfg_clone = cfg.clone();
    cfg_clone.is_a_on_device = input.is_on_device();
    cfg_clone.is_result_on_device = output.is_on_device();

    <<Zq as FieldImpl>::Config as RnsConversion<Zq, ZqRns>>::to_rns(input, cfg, output)
}

pub fn from_rns<Zq: FieldImpl, ZqRns: FieldImpl>(
    input: &(impl HostOrDeviceSlice<ZqRns> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<Zq> + ?Sized),
) -> Result<(), eIcicleError>
where
    <Zq as FieldImpl>::Config: RnsConversion<Zq, ZqRns>,
{
    if input.len() != output.len() {
        eprintln!(
            "input (size={}) and output (size={}) slices must have the same length",
            input.len(),
            output.len()
        );
        return Err(eIcicleError::InvalidArgument);
    }

    // check device slices are on active device
    if input.is_on_device() && !input.is_on_active_device() {
        eprintln!("input not allocated on an inactive device");
        return Err(eIcicleError::InvalidArgument);
    }
    if output.is_on_device() && !output.is_on_active_device() {
        eprintln!("output not allocated on an inactive device");
        return Err(eIcicleError::InvalidArgument);
    }

    let mut cfg_clone = cfg.clone();
    cfg_clone.is_a_on_device = input.is_on_device();
    cfg_clone.is_result_on_device = output.is_on_device();

    <<Zq as FieldImpl>::Config as RnsConversion<Zq, ZqRns>>::from_rns(input, cfg, output)
}

// (3) implement a macro that implements RNS-conversions for a given FieldConfig type, via C bindings

// (4) implement generic tests for the RNS-conversions (to_rns, from_rns)
