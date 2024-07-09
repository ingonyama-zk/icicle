use icicle_core::error::IcicleResult;
use icicle_core::hash::SpongeHash;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::tree::{mmcs::FieldMmcs, TreeBuilderConfig};
use icicle_core::{impl_mmcs, Matrix};
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};
use std::ffi::c_void;

use crate::curve::ScalarField;

impl_mmcs!("bn254", bn254_mmcs, ScalarField, ScalarCfg, Bn254Mmcs);
