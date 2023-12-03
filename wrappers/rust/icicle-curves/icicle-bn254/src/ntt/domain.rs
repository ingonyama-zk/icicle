use icicle_core::ntt::{Butterfly, Decimation, NTTConfigCuda, Ordering};
use icicle_cuda_runtime::device_context::{get_default_device_context, DeviceContext};
use icicle_cuda_runtime::memory::DeviceSlice;
use std::default;

pub(super) type ECNTTDomain<'a> = Domain<'a, G1Projective, ScalarField>;
pub(super) type NTTDomain<'a> = Domain<'a, ScalarField, ScalarField>;

use crate::curve::*;

use super::{config::*, ntt_internal};

/// Represents the NTT domain
pub struct Domain<'a, E, S> {
    config: NTTConfigCuda<'a, E, S>,
}

impl<'a, E, S> Domain<'a, E, S> {
    pub fn new(size: usize, ctx: DeviceContext<'a>) -> Self {
        Domain {
            config: get_ntt_config(size, ctx),
        }
    }

    pub fn get_output_on_device(&self) -> Result<*mut E, &'static str> {
        if self
            .config
            .is_output_on_device
        {
            Ok(self
                .config
                .inout)
        } else {
            Err("Output should be on device.")
        }
    }

    pub fn get_input_on_device(&self) -> Result<*mut E, &'static str> {
        if self
            .config
            .is_input_on_device
        {
            Ok(self
                .config
                .inout)
        } else {
            Err("Input should be on device.")
        }
    }

    pub fn get_input(&self) -> Result<*mut E, &'static str> {
        if !self
            .config
            .is_input_on_device
        {
            Ok(self
                .config
                .inout)
        } else {
            Err("Output is on device.")
        }
    }

    pub fn get_output(&self) -> Result<*mut E, &'static str> {
        if !self
            .config
            .is_output_on_device
        {
            Ok(self
                .config
                .inout)
        } else {
            Err("Output is on device.")
        }
    }

    pub(crate) fn new_for_default_context(size: usize) -> Self {
        let ctx = get_default_device_context();
        // let default_root_of_unity = S::default(); //TODO: implement
        let domain = Domain::new(size, ctx);
        domain
    }
}

// Add implementations for other methods and structs as needed.

impl<'a, E: 'static, S: 'static> Domain<'a, E, S> {
    // ... previous methods ...

    // NTT methods
    pub fn ntt(&mut self, inout: &mut [E]) {
        let batch_size = 1;

        let size = inout.len();

        if size
            != self
                .config
                .size as _
        {
            //TODO: test for this error
            panic!(
                "input lenght: {} does not match domain size: {}",
                size,
                self.config
                    .size
            )
        }

        self.config
            .inout = inout.as_mut_ptr(); // as *mut _ as *mut E;
        self.config
            .is_inverse = false;
        self.config
            .is_input_on_device = false;
        self.config
            .is_output_on_device = false;
        // self.config
        //     .ordering = Ordering::default(); //TODO: each call?
        self.config
            .batch_size = batch_size as i32;

        ntt_internal(&mut self.config);
    }

    pub fn ntt_on_device(&mut self, inout: &mut DeviceSlice<E>) {
        // Implementation for NTT on device
    }

    pub fn ntt_batch(&mut self, inout: &mut [E]) {
        // Implementation for batched NTT
    }

    pub fn ntt_batch_on_device(&mut self, inout: &mut DeviceSlice<E>) {
        // Implementation for batched NTT on device
    }

    pub fn ntt_coset(&mut self, inout: &mut [E], coset: &mut [E]) {
        // Implementation for NTT with coset
    }

    pub fn ntt_coset_on_device(&mut self, inout: &mut DeviceSlice<E>, coset: &mut DeviceSlice<E>) {
        // Implementation for NTT with coset on device
    }

    pub fn ntt_coset_batch(&mut self, inout: &mut [E], coset: &mut [E]) {
        // Implementation for batched NTT with coset
    }

    pub fn ntt_coset_batch_on_device(&mut self, inout: &mut DeviceSlice<E>, coset: &mut DeviceSlice<E>) {
        // Implementation for batched NTT with coset on device
    }

    // iNTT methods
    pub fn intt(&mut self, inout: &mut [E]) {
        // Implementation for iNTT
    }

    pub fn intt_on_device(&mut self, inout: &mut DeviceSlice<E>) {
        // Implementation for iNTT on device
    }

    pub fn intt_batch(&mut self, inout: &mut [E]) {
        // Implementation for batched iNTT
    }

    pub fn intt_batch_on_device(&mut self, inout: &mut DeviceSlice<E>) {
        // Implementation for batched iNTT on device
    }

    pub fn intt_coset(&mut self, inout: &mut [E], coset: &mut [E]) {
        // Implementation for iNTT with coset
    }

    pub fn intt_coset_on_device(&mut self, inout: &mut DeviceSlice<E>, coset: &mut DeviceSlice<E>) {
        // Implementation for iNTT with coset on device
    }

    pub fn intt_coset_batch(&mut self, inout: &mut [E], coset: &mut [E]) {
        // Implementation for batched iNTT with coset
    }

    pub fn intt_coset_batch_on_device(&mut self, inout: &mut DeviceSlice<E>, coset: &mut DeviceSlice<E>) {
        // Implementation for batched iNTT with coset on device
    }

    // Ordering setter
    pub fn set_ordering(&mut self, ordering: Ordering) {
        self.config
            .ordering = ordering;
    }
}
