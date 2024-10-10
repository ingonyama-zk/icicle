use crate::hash::{Hasher, HasherHandle};
use icicle_runtime::{
    config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStreamHandle,
};
use std::{ffi::c_void, mem, ptr, slice};

/// Enum representing the padding policy when the input is smaller than expected by the tree structure.
#[repr(C)]
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
#[repr(C)]
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
    fn icicle_merkle_proof_delete(proof: MerkleProofHandle) -> eIcicleError;
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

type MerkleTreeHandle = *const c_void;

pub struct MerkleTree {
    handle: MerkleTreeHandle,
}

// External C functions
extern "C" {
    fn icicle_merkle_tree_create(
        layer_hashes: *const MerkleTreeHandle, // expecting c-style-array of those
        layer_hashes_len: u64,
        leaf_element_size: u64,
        output_store_min_layer: u64,
    ) -> MerkleTreeHandle;

    fn icicle_merkle_tree_delete(tree: MerkleTreeHandle);

    fn icicle_merkle_tree_build(
        tree: MerkleTreeHandle,
        leaves: *const u8,
        size: u64,
        config: *const MerkleTreeConfig,
    ) -> eIcicleError;

    fn icicle_merkle_tree_get_root(tree: MerkleTreeHandle, out_size: *mut u64) -> *const u8;

    fn icicle_merkle_tree_get_proof(
        tree: MerkleTreeHandle,
        leaves: *const u8,
        size: u64,
        leaf_idx: u64,
        is_pruned: bool,
        config: *const MerkleTreeConfig,
        merkle_proof: MerkleProofHandle,
    ) -> eIcicleError;

    fn icicle_merkle_tree_verify(
        tree: MerkleTreeHandle,
        merkle_proof: MerkleProofHandle,
        valid: *mut bool,
    ) -> eIcicleError;
}

impl MerkleTree {
    // Create a new MerkleTree with an array/vector of Hasher structs for the layer hashes
    pub fn new(
        layer_hashes: &[&Hasher],
        leaf_element_size: u64,
        output_store_min_layer: u64,
    ) -> Result<Self, eIcicleError> {
        unsafe {
            // Collect the Hasher handles from the Hasher structs
            let hash_handles: Vec<HasherHandle> = layer_hashes
                .iter()
                .map(|h| h.handle)
                .collect();

            let handle = icicle_merkle_tree_create(
                hash_handles.as_ptr(),
                hash_handles.len() as u64,
                leaf_element_size as u64,
                output_store_min_layer as u64,
            );

            if handle.is_null() {
                Err(eIcicleError::UnknownError)
            } else {
                Ok(MerkleTree { handle })
            }
        }
    }

    // Templated function to build the Merkle tree using any type of leaves
    pub fn build<T>(
        &self,
        leaves: &(impl HostOrDeviceSlice<T> + ?Sized),
        cfg: &MerkleTreeConfig,
    ) -> Result<(), eIcicleError> {
        // check device slices are on active device
        if leaves.is_on_device() && !leaves.is_on_active_device() {
            eprintln!("leaves not allocated on the active device");
            return Err(eIcicleError::InvalidPointer);
        }

        let mut local_cfg = cfg.clone();
        local_cfg.is_leaves_on_device = leaves.is_on_device();

        let byte_size = (leaves.len() * std::mem::size_of::<T>()) as u64;
        unsafe { icicle_merkle_tree_build(self.handle, leaves.as_ptr() as *const u8, byte_size, &local_cfg).wrap() }
    }

    // Templated function to get the Merkle root as a slice of type T
    pub fn get_root<T>(&self) -> Result<&[T], eIcicleError> {
        let mut size: u64 = 0;
        let root_ptr = unsafe { icicle_merkle_tree_get_root(self.handle, &mut size) };

        if root_ptr.is_null() {
            Err(eIcicleError::UnknownError)
        } else {
            let element_size = std::mem::size_of::<T>() as usize;
            let num_elements = size as usize / element_size;
            unsafe { Ok(slice::from_raw_parts(root_ptr as *const T, num_elements)) }
        }
    }

    // Templated function to retrieve a Merkle proof for a specific element of type T
    pub fn get_proof<T>(
        &self,
        leaves: &(impl HostOrDeviceSlice<T> + ?Sized),
        leaf_idx: u64,
        config: &MerkleTreeConfig,
    ) -> Result<MerkleProof, eIcicleError> {
        // check device slices are on active device
        if leaves.is_on_device() && !leaves.is_on_active_device() {
            eprintln!("leaves not allocated on the active device");
            return Err(eIcicleError::InvalidPointer);
        }

        let proof = MerkleProof::new().unwrap();
        let byte_size = (leaves.len() * std::mem::size_of::<T>()) as u64;
        let result = unsafe {
            icicle_merkle_tree_get_proof(
                self.handle,
                leaves.as_ptr() as *const u8,
                byte_size,
                leaf_idx as u64,
                false,
                config,
                proof.handle,
            )
        };

        if result == eIcicleError::Success {
            Ok(proof)
        } else {
            Err(result)
        }
    }

    pub fn verify(&self, proof: &MerkleProof) -> Result<bool, eIcicleError> {
        let mut verification_valid: bool = false;
        let result = unsafe { icicle_merkle_tree_verify(self.handle, proof.handle, &mut verification_valid) };
        if result == eIcicleError::Success {
            Ok(verification_valid)
        } else {
            Err(result)
        }
    }
}

impl Drop for MerkleTree {
    fn drop(&mut self) {
        unsafe {
            icicle_merkle_tree_delete(self.handle);
        }
    }
}
