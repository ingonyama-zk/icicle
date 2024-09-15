#pragma once

#include <memory>
#include <cstddef>
#include <vector>
#include <iostream> // For streams
#include <stdexcept>
#include "icicle/runtime.h"

namespace icicle {

  /**
   * @brief Class representing the Merkle path in a move-only manner.
   *
   * This class manages a Merkle path as a collection of bytes. It is designed to be move-only,
   * meaning it can be transferred but not copied, ensuring clear ownership of the path data.
   * The path is stored using a `std::vector<std::byte>` for easy management and flexibility.
   */
  class MerkleProof
  {
  public:
    /**
     * @brief Simple constructor for MerkleProof.
     */
    explicit MerkleProof() = default;

    /**
     * @brief Allocate and copy leaf and root data using raw byte pointers.
     * @param pruned_path Whether the Merkle path is pruned.
     * @param leaf_idx The index of the leaf for which the path is a proof.
     * @param leaf Pointer to the leaf data as std::byte*. Can be host/device memory.
     * @param leaf_size The size of the leaf data.
     * @param root Pointer to the root data as std::byte*. Can be host/device memory.
     * @param root_size The size of the root data.
     */
    void allocate_from_bytes(
      bool pruned_path,
      uint64_t leaf_idx,
      const std::byte* leaf,
      std::size_t leaf_size,
      const std::byte* root,
      std::size_t root_size)
    {
      m_pruned = pruned_path;
      m_leaf_index = leaf_idx;

      if (root != nullptr && root_size > 0) {
        m_root.resize(root_size);
        ICICLE_CHECK(icicle_copy(m_root.data(), root, root_size));
      }

      if (leaf != nullptr && leaf_size > 0) {
        m_leaf.resize(leaf_size);
        ICICLE_CHECK(icicle_copy(m_leaf.data(), leaf, leaf_size));
      }
    }

    /**
     * @brief Allocate and copy leaf and root data using templated types.
     * @tparam LEAF The type of the leaf data.
     * @tparam DIGEST The type of the root data.
     * @param pruned_path Whether the Merkle path is pruned.
     * @param leaf_idx The index of the leaf for which the path is a proof.
     * @param leaf The leaf data.
     * @param root The root data.
     */
    template <typename LEAF, typename DIGEST>
    void allocate(bool pruned_path, uint64_t leaf_idx, const LEAF& leaf, const DIGEST& root)
    {
      m_pruned = pruned_path;
      m_leaf_index = leaf_idx;

      // Allocate and copy root data
      m_root.resize(sizeof(DIGEST));
      ICICLE_CHECK(icicle_copy(m_root.data(), &root, sizeof(DIGEST)));

      // Allocate and copy leaf data
      m_leaf.resize(sizeof(LEAF));
      ICICLE_CHECK(icicle_copy(m_leaf.data(), &leaf, sizeof(LEAF)));
    }

    /**
     * @brief Check if the Merkle path is pruned.
     * @return True if the path is pruned, false otherwise.
     */
    bool is_pruned() const { return m_pruned; }

    /**
     * @brief Get a pointer to the path data.
     * @return Pointer to the path data.
     */
    const std::byte* get_path() const { return m_path.data(); }

    /**
     * @brief Get the size of the path data.
     * @return The size of the path data.
     */
    uint64_t get_path_size() const { return m_path.size(); }

    /**
     * @brief Push a node to the path, given as bytes, using icicle_copy.
     * @param node The pointer to the node data.
     * @param size The size of the node data in bytes.
     */
    void push_node_to_path(const std::byte* node, uint64_t size)
    {
      std::size_t old_size = m_path.size();
      m_path.resize(old_size + size);
      ICICLE_CHECK(icicle_copy(m_path.data() + old_size, node, size));
    }

    /**
     * @brief Push a node to the path, given as a typed object.
     * @tparam T The type of the node.
     * @param node The pointer to the node.
     */
    template <typename T>
    void push_node_to_path(const T& node)
    {
      push_node_to_path(reinterpret_cast<const std::byte*>(&node), sizeof(T));
    }

    /**
     * @brief Access data at a specific offset and cast it to the desired type.
     * @tparam T The type to cast the data to.
     * @param offset The byte offset to access.
     * @return Pointer to the data cast to type T.
     */
    template <typename T>
    const T* access_path_at_offset(uint64_t offset)
    {
      if (offset >= m_path.size()) { throw std::out_of_range("Offset out of bounds"); }
      return reinterpret_cast<const T*>(m_path.data() + offset);
    }

    /**
     * @brief Get the index of the leaf this path is a proof for.
     * @return Index of the proved leaf.
     */
    uint64_t get_leaf_idx() const { return m_leaf_index; }

    /**
     * @brief Get a pointer to the leaf data.
     * @return Pointer to the leaf data, or nullptr if no leaf data is available.
     */
    const std::byte* get_leaf() const { return m_leaf.empty() ? nullptr : m_leaf.data(); }

    /**
     * @brief Get the size of the leaf data.
     * @return The size of the leaf data, or 0 if no leaf data is available.
     */
    uint64_t get_leaf_size() const { return m_leaf.size(); }

    /**
     * @brief Get a pointer to the root data.
     * @return Pointer to the root data, or nullptr if no root data is available.
     */
    const std::byte* get_root() const { return m_root.empty() ? nullptr : m_root.data(); }

    /**
     * @brief Get the size of the root data.
     * @return The size of the root data, or 0 if no root data is available.
     */
    uint64_t get_root_size() const { return m_root.size(); }

  private:
    bool m_pruned{false};          ///< Whether the Merkle path is pruned.
    uint64_t m_leaf_index{0};      ///< Index of the leaf this path is a proof for.
    std::vector<std::byte> m_leaf; ///< Optional leaf data.
    std::vector<std::byte> m_root; ///< Optional root data.
    std::vector<std::byte> m_path; ///< Path data.
  };

} // namespace icicle