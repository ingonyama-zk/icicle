use icicle_runtime::{
    config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStreamHandle,
};
use std::{ffi::c_void, mem, ptr, slice};

/// Enum representing the padding policy when the input is smaller than expected by the tree structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaddingPolicy {
    None,        // No padding, assume input is correctly sized.
    ZeroPadding, // Pad the input with zeroes to fit the expected input size.
    LastValue,   // Pad the input by repeating the last value.
}

/// Configuration structure for Merkle tree operations.
///
/// This structure holds the configuration options for Merkle tree operations, including tree construction,
/// path computation, and verification. It allows specifying whether the data (leaves, tree, and paths)
/// reside on the device (e.g., GPU) or the host (e.g., CPU), and supports both synchronous and asynchronous
/// execution modes, as well as backend-specific extensions. It also provides a padding policy for handling
/// cases where the input size is smaller than expected by the tree structure.
#[derive(Clone)]
pub struct MerkleTreeConfig {
    pub stream_handle: IcicleStreamHandle, // Stream for asynchronous execution. Default is null for synchronous execution.
    pub is_leaves_on_device: bool,         // True if leaves are on the device (GPU), false if on the host (CPU).
    pub is_tree_on_device: bool, // True if the tree results are allocated on the device (GPU), false if on the host (CPU).
    pub is_async: bool,          // True for asynchronous execution, false for synchronous.
    pub padding_policy: PaddingPolicy, // Policy for handling cases where the input is smaller than expected.
    pub ext: ConfigExtension,    // Backend-specific extensions for advanced configurations.
}

impl MerkleTreeConfig {
    /// Generates a default configuration for Merkle tree operations.
    ///
    /// This function provides a default configuration for Merkle tree operations with synchronous execution
    /// and all data (leaves, tree results, and paths) residing on the host (CPU).
    pub fn default() -> Self {
        Self {
            stream_handle: ptr::null_mut(),      // Default stream handle (synchronous).
            is_leaves_on_device: false,          // Default: leaves on host (CPU).
            is_tree_on_device: false,            // Default: tree results on host (CPU).
            is_async: false,                     // Default: synchronous execution.
            padding_policy: PaddingPolicy::None, // Default: no padding.
            ext: ConfigExtension::new(),         // Default: no backend-specific extensions.
        }
    }
}

type MerkleProofHandle = *const c_void;

pub struct MerkleProof {
    handle: MerkleProofHandle,
}

// External C functions for merkle proof
extern "C" {
    fn icicle_merkle_proof_create() -> MerkleProofHandle;
    fn icicle_merkle_proof_delete(proof: MerkleProofHandle) -> i32;
    fn icicle_merkle_proof_is_pruned(proof: MerkleProofHandle) -> bool;
    fn icicle_merkle_proof_get_path(proof: MerkleProofHandle, out_size: *mut usize) -> *const u8;
    fn icicle_merkle_proof_get_leaf(
        proof: MerkleProofHandle,
        out_size: *mut usize,
        out_leaf_idx: *mut u64,
    ) -> *const u8;
    fn icicle_merkle_proof_get_root(proof: MerkleProofHandle, out_size: *mut usize) -> *const u8;
}

impl MerkleProof {
    /// Create a new MerkleProof object.
    pub fn new() -> Result<Self, eIcicleError> {
        unsafe {
            let handle = icicle_merkle_proof_create();
            if handle.is_null() {
                Err(eIcicleError::AllocationFailed)
            } else {
                Ok(MerkleProof { handle })
            }
        }
    }

    /// Check if the Merkle path is pruned.
    pub fn is_pruned(&self) -> bool {
        unsafe { icicle_merkle_proof_is_pruned(self.handle) }
    }

    /// Get the path data as a slice of type `T`.
    pub fn get_path<T>(&self) -> &[T] {
        let mut size = 0;
        unsafe {
            let ptr = icicle_merkle_proof_get_path(self.handle, &mut size);
            if ptr.is_null() {
                &[]
            } else {
                // Calculate how many `T` elements fit into the byte buffer
                let element_count = size / mem::size_of::<T>();
                slice::from_raw_parts(ptr as *const T, element_count)
            }
        }
    }

    /// Get the leaf data as a slice of type `T` and its index.
    pub fn get_leaf<T>(&self) -> (&[T], u64) {
        let mut size = 0;
        let mut leaf_idx = 0;
        unsafe {
            let ptr = icicle_merkle_proof_get_leaf(self.handle, &mut size, &mut leaf_idx);
            if ptr.is_null() {
                (&[], 0)
            } else {
                // Calculate how many `T` elements fit into the byte buffer
                let element_count = size / mem::size_of::<T>();
                (slice::from_raw_parts(ptr as *const T, element_count), leaf_idx)
            }
        }
    }

    /// Get the root data as a slice of type `T`.
    pub fn get_root<T>(&self) -> &[T] {
        let mut size = 0;
        unsafe {
            let ptr = icicle_merkle_proof_get_root(self.handle, &mut size);
            if ptr.is_null() {
                &[]
            } else {
                // Calculate how many `T` elements fit into the byte buffer
                let element_count = size / mem::size_of::<T>();
                slice::from_raw_parts(ptr as *const T, element_count)
            }
        }
    }
}

impl Drop for MerkleProof {
    fn drop(&mut self) {
        unsafe {
            if !self
                .handle
                .is_null()
            {
                let _ = icicle_merkle_proof_delete(self.handle);
            }
        }
    }
}
