use crate::field::{ScalarCfg, ScalarField};

use icicle_core::impl_program_field;
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_program_field!("stark252", stark252, ScalarField, ScalarCfg);
// impl_vec_ops_field!("stark252", stark252, ScalarField);