use crate::{
    hash::{Hasher, HasherHandle},
    traits::Handle,
};
use icicle_runtime::{
    config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStreamHandle,
};
use serde::{
    de::{self, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::{ffi::c_void, fmt, mem, ptr, slice};

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
    is_leaves_on_device: bool,             // True if leaves are on the device (GPU), false if on the host (CPU).
    is_tree_on_device: bool, // True if the tree results are allocated on the device (GPU), false if on the host (CPU).
    pub is_async: bool,      // True for asynchronous execution, false for synchronous.
    pub padding_policy: PaddingPolicy, // Policy for handling cases where the input is smaller than expected.
    pub ext: ConfigExtension, // Backend-specific extensions for advanced configurations.
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

pub struct MerkleProofData<T> {
    pub is_pruned: bool,
    pub leaf_idx: u64,
    pub leaf: Vec<T>,
    pub root: Vec<T>,
    pub path: Vec<T>,
}

impl<T> MerkleProofData<T> {
    pub fn new(is_pruned: bool, leaf_idx: u64, leaf: Vec<T>, root: Vec<T>, path: Vec<T>) -> Self {
        Self {
            is_pruned,
            leaf_idx,
            leaf,
            root,
            path,
        }
    }
}

impl<T: Clone> From<&MerkleProof> for MerkleProofData<T> {
    fn from(proof: &MerkleProof) -> Self {
        let (leaf, leaf_idx) = proof.get_leaf::<T>();
        let root = proof.get_root::<T>();
        let path = proof.get_path::<T>();
        Self::new(proof.is_pruned(), leaf_idx, leaf.to_vec(), root.to_vec(), path.to_vec())
    }
}

pub type MerkleProofHandle = *const c_void;

pub struct MerkleProof {
    handle: MerkleProofHandle,
}

// External C functions for merkle proof
extern "C" {
    fn icicle_merkle_proof_create() -> MerkleProofHandle;
    fn icicle_merkle_proof_create_with_data(
        pruned_path: bool,
        leaf_idx: u64,
        leaf: *const u8,
        leaf_size: usize,
        root: *const u8,
        root_size: usize,
        path: *const u8,
        path_size: usize,
    ) -> MerkleProofHandle;
    fn icicle_merkle_proof_delete(proof: MerkleProofHandle) -> eIcicleError;
    fn icicle_merkle_proof_is_pruned(proof: MerkleProofHandle) -> bool;
    fn icicle_merkle_proof_get_path(proof: MerkleProofHandle, out_size: *mut usize) -> *const u8;
    fn icicle_merkle_proof_get_leaf(
        proof: MerkleProofHandle,
        out_size: *mut usize,
        out_leaf_idx: *mut u64,
    ) -> *const u8;
    fn icicle_merkle_proof_get_root(proof: MerkleProofHandle, out_size: *mut usize) -> *const u8;
    fn icicle_merkle_proof_get_serialized_size(proof: MerkleProofHandle, out_size: *mut usize) -> eIcicleError;
    fn icicle_merkle_proof_serialize(proof: MerkleProofHandle, buffer: *mut u8, size: usize) -> eIcicleError;
    fn icicle_merkle_proof_deserialize(proof: *mut MerkleProofHandle, buffer: *const u8, size: usize) -> eIcicleError;
    fn icicle_merkle_proof_serialize_to_file(
        proof: MerkleProofHandle,
        filename: *const u8,
        filename_len: usize,
    ) -> eIcicleError;
    fn icicle_merkle_proof_deserialize_from_file(
        proof: *mut MerkleProofHandle,
        filename: *const u8,
        filename_len: usize,
    ) -> eIcicleError;
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

    /// Create a new MerkleProof object with specified leaf, root, and path data.
    pub fn create_with_data<T>(
        is_pruned: bool,
        leaf_idx: u64,
        leaf: &[T],
        root: &[T],
        path: &[T],
    ) -> Result<Self, eIcicleError> {
        let leaf_bytes = unsafe { slice::from_raw_parts(leaf.as_ptr() as *const u8, leaf.len() * mem::size_of::<T>()) };
        let root_bytes = unsafe { slice::from_raw_parts(root.as_ptr() as *const u8, root.len() * mem::size_of::<T>()) };
        let path_bytes = unsafe { slice::from_raw_parts(path.as_ptr() as *const u8, path.len() * mem::size_of::<T>()) };

        unsafe {
            let handle = icicle_merkle_proof_create_with_data(
                is_pruned,
                leaf_idx,
                leaf_bytes.as_ptr(),
                leaf_bytes.len(),
                root_bytes.as_ptr(),
                root_bytes.len(),
                path_bytes.as_ptr(),
                path_bytes.len(),
            );
            if handle.is_null() {
                Err(eIcicleError::UnknownError)
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

    pub unsafe fn from_handle(handle: MerkleProofHandle) -> Self {
        Self { handle }
    }
}

impl Serialize for MerkleProof {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut size = 0;
        unsafe {
            icicle_merkle_proof_get_serialized_size(self.handle, &mut size)
                .wrap_value(size)
                .map_err(serde::ser::Error::custom)?;
            let mut buffer = vec![0u8; size];
            icicle_merkle_proof_serialize(self.handle, buffer.as_mut_ptr(), buffer.len())
                .wrap_value(buffer)
                .map_err(serde::ser::Error::custom)
                .and_then(|b| serializer.serialize_bytes(&b))
        }
    }
}

impl<'de> Deserialize<'de> for MerkleProof {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MerkleProofVisitor;

        impl<'de> Visitor<'de> for MerkleProofVisitor {
            type Value = MerkleProof;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a byte array representing a MerkleProof")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let mut handle = ptr::null();
                unsafe {
                    icicle_merkle_proof_deserialize(&mut handle, v.as_ptr(), v.len())
                        .wrap_value(MerkleProof { handle })
                        .map_err(de::Error::custom)
                }
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut buffer = Vec::with_capacity(
                    seq.size_hint()
                        .unwrap_or(0),
                );
                while let Some(byte) = seq.next_element::<u8>()? {
                    buffer.push(byte);
                }
                self.visit_bytes(&buffer)
            }
        }

        deserializer.deserialize_bytes(MerkleProofVisitor)
    }
}

impl Handle for MerkleProof {
    fn handle(&self) -> *const c_void {
        self.handle
    }
}

impl<T> TryFrom<MerkleProofData<T>> for MerkleProof {
    type Error = eIcicleError;
    fn try_from(data: MerkleProofData<T>) -> Result<Self, Self::Error> {
        Self::create_with_data(data.is_pruned, data.leaf_idx, &data.leaf, &data.root, &data.path)
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
        layer_hashes: *const HasherHandle, // expecting c-style-array of those
        layer_hashes_len: u64,
        leaf_element_size: u64,
        output_store_min_layer: u64,
    ) -> MerkleTreeHandle;

    fn icicle_merkle_tree_delete(tree: MerkleTreeHandle) -> eIcicleError;

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
        pruned_path: bool,
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
                pruned_path,
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
            if !self
                .handle
                .is_null()
            {
                let _ = icicle_merkle_tree_delete(self.handle);
            }
        }
    }
}
