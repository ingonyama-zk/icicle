use crate::cuda::*;

use super::config::*;

/// Represents the NTT domain
pub struct Domain<E, S> {
    config: NTTConfigCuda<E, S>,
}

impl<E, S> Domain<E, S> {
    pub fn new(size: usize, root_of_unity: S, ctx: DeviceContext) -> Self {
        Domain {
            config: get_ntt_config(size, root_of_unity, ctx),
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
}

// Add implementations for other methods and structs as needed.

impl<E, S> Domain<E, S> {
    // ... previous methods ...

    // NTT methods
    pub fn ntt(&mut self, inout: &mut [E]) {
        // Implementation for NTT
        todo!()
    }

    pub fn ntt_on_device(&mut self, inout: &mut DevicePointer<E>) {
        // Implementation for NTT on device
    }

    pub fn ntt_batch(&mut self, inout: &mut [E]) {
        // Implementation for batched NTT
    }

    pub fn ntt_batch_on_device(&mut self, inout: &mut DevicePointer<E>) {
        // Implementation for batched NTT on device
    }

    pub fn ntt_coset(&mut self, inout: &mut [E], coset: &mut [E]) {
        // Implementation for NTT with coset
    }

    pub fn ntt_coset_on_device(&mut self, inout: &mut DevicePointer<E>, coset: &mut DevicePointer<E>) {
        // Implementation for NTT with coset on device
    }

    pub fn ntt_coset_batch(&mut self, inout: &mut [E], coset: &mut [E]) {
        // Implementation for batched NTT with coset
    }

    pub fn ntt_coset_batch_on_device(&mut self, inout: &mut DevicePointer<E>, coset: &mut DevicePointer<E>) {
        // Implementation for batched NTT with coset on device
    }

    // iNTT methods
    pub fn intt(&mut self, inout: &mut [E]) {
        // Implementation for iNTT
    }

    pub fn intt_on_device(&mut self, inout: &mut DevicePointer<E>) {
        // Implementation for iNTT on device
    }

    pub fn intt_batch(&mut self, inout: &mut [E]) {
        // Implementation for batched iNTT
    }

    pub fn intt_batch_on_device(&mut self, inout: &mut DevicePointer<E>) {
        // Implementation for batched iNTT on device
    }

    pub fn intt_coset(&mut self, inout: &mut [E], coset: &mut [E]) {
        // Implementation for iNTT with coset
    }

    pub fn intt_coset_on_device(&mut self, inout: &mut DevicePointer<E>, coset: &mut DevicePointer<E>) {
        // Implementation for iNTT with coset on device
    }

    pub fn intt_coset_batch(&mut self, inout: &mut [E], coset: &mut [E]) {
        // Implementation for batched iNTT with coset
    }

    pub fn intt_coset_batch_on_device(&mut self, inout: &mut DevicePointer<E>, coset: &mut DevicePointer<E>) {
        // Implementation for batched iNTT with coset on device
    }

    // Ordering setter
    pub fn set_ordering(&mut self, ordering: Ordering) {
        self.config
            .ordering = ordering;
    }
}
