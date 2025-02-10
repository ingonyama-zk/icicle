use crate::field::{ScalarCfg, ScalarField};

use icicle_core::gate_ops::{GateOps, GateOpsConfig};
use icicle_core::{impl_gate_ops_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_gate_ops_field!("koalabear", koalabear, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::gate_ops::tests::*;
    use icicle_core::impl_gate_ops_tests;

    impl_gate_ops_tests!(ScalarField);
}
