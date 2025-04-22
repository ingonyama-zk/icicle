#pragma once

#include "icicle/merkle/merkle_proof.h"
#include "icicle/serialization.h"

namespace icicle {
  template <>
  struct BinarySerializeImpl<MerkleProof> {
    static eIcicleError serialized_size(const MerkleProof& obj, size_t& size)
    {
      size = sizeof(bool);      // pruned
      size += sizeof(uint64_t); // leaf_index
      size += sizeof(size_t);   // leaf_size
      auto [leaf, leaf_size, leaf_index] = obj.get_leaf();
      size += leaf_size * sizeof(std::byte); // leaf
      size += sizeof(size_t);                // root_size
      auto [root, root_size] = obj.get_root();
      size += root_size * sizeof(std::byte); // root
      size += sizeof(size_t);                // path_size
      auto [path, path_size] = obj.get_path();
      size += path_size * sizeof(std::byte); // path
      return eIcicleError::SUCCESS;
    }
    static eIcicleError pack_and_advance(std::byte*& buffer, size_t& buffer_length, const MerkleProof& obj)
    {
      bool pruned = obj.is_pruned();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &pruned, sizeof(bool)));

      auto [leaf, leaf_size, leaf_index] = obj.get_leaf();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &leaf_index, sizeof(uint64_t)));

      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &leaf_size, sizeof(size_t)));
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, leaf, leaf_size * sizeof(std::byte)));

      auto [root, root_size] = obj.get_root();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &root_size, sizeof(size_t)));
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, root, root_size * sizeof(std::byte)));

      auto [path, path_size] = obj.get_path();
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, &path_size, sizeof(size_t)));
      ICICLE_CHECK_IF_RETURN(memcpy_shift_destination(buffer, buffer_length, path, path_size * sizeof(std::byte)));

      return eIcicleError::SUCCESS;
    }
    static eIcicleError unpack_and_advance(const std::byte*& buffer, size_t& buffer_length, MerkleProof& obj)
    {
      bool pruned;
      int64_t leaf_idx;
      std::vector<std::byte> leaf;
      std::vector<std::byte> root;
      std::vector<std::byte> path;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&pruned, buffer_length, buffer, sizeof(bool)));

      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&leaf_idx, buffer_length, buffer, sizeof(uint64_t)));

      size_t leaf_size;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&leaf_size, buffer_length, buffer, sizeof(size_t)));
      leaf.resize(leaf_size);
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(leaf.data(), buffer_length, buffer, leaf_size * sizeof(std::byte)));

      size_t root_size;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&root_size, buffer_length, buffer, sizeof(size_t)));
      root.resize(root_size);
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(root.data(), buffer_length, buffer, root_size * sizeof(std::byte)));

      size_t path_size;
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(&path_size, buffer_length, buffer, sizeof(size_t)));
      path.resize(path_size);
      ICICLE_CHECK_IF_RETURN(memcpy_shift_source(path.data(), buffer_length, buffer, path_size * sizeof(std::byte)));

      MerkleProof proof = MerkleProof(pruned, leaf_idx, std::move(leaf), std::move(root), std::move(path));
      obj = std::move(proof);

      return eIcicleError::SUCCESS;
    }
  };
} // namespace icicle
