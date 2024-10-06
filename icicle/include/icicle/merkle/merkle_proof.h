#pragma once

#include <memory>
#include <cstddef>
#include <vector>
#include <iostream> // For streams
#include <stdexcept>
#include <utility> // For std::pair
#include "icicle/runtime.h"

namespace icicle {

  /**
   * @brief Represents a Merkle proof with leaf, root, and path data.
   *
   * This class encapsulates the Merkle proof, managing the leaf, root, and path as byte arrays.
   * It provides functionality to allocate, copy, and access these components, supporting both
   * raw byte manipulation and type-safe access via templates.
   */
  class MerkleProof
  {
  public:
    explicit MerkleProof() = default;

    /**
     * @brief Allocates memory for the Merkle proof and copies the leaf and root data.
     *
     * This function initializes the Merkle proof by setting whether the path is pruned and storing
     * the index of the leaf being proved. It then allocates memory for both the root and leaf data
     * and copies the provided data into the allocated buffers. It assumes the provided pointers
     * (leaf and root) point to either host memory or memory allocated via `icicle_malloc`.
     *
     * @param pruned_path Whether the Merkle path is pruned.
     * @param leaf_idx The index of the leaf for which this is a proof.
     * @param leaf Pointer to the leaf data as a sequence of bytes. It can be host memory or memory allocated via
     * `icicle_malloc`.
     * @param leaf_size The size of the leaf data in bytes.
     * @param root Pointer to the root data as a sequence of bytes. It can be host memory or memory allocated via
     * `icicle_malloc`.
     * @param root_size The size of the root data in bytes.
     */
    void allocate(
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
        // Note: assuming root is either host memory or allocated via icicle_malloc!
        ICICLE_CHECK(icicle_copy(m_root.data(), root, root_size));
      }

      if (leaf != nullptr && leaf_size > 0) {
        m_leaf.resize(leaf_size);
        // Note: assuming leaf is either host memory or allocated via icicle_malloc!
        ICICLE_CHECK(icicle_copy(m_leaf.data(), leaf, leaf_size));
      }
    }

    /**
     * @brief Check if the Merkle path is pruned.
     * @return True if the path is pruned, false otherwise.
     */
    bool is_pruned() const { return m_pruned; }

    /**
     * @brief Returns a pair containing the pointer to the path data and its size.
     * @return A pair of (path data pointer, path size).
     */
    std::pair<const std::byte*, std::size_t> get_path() const
    {
      return {m_path.empty() ? nullptr : m_path.data(), m_path.size()};
    }

    /**
     * @brief Returns a tuple containing the pointer to the leaf data, its size and index.
     * @return A tuple of (leaf data pointer, leaf size, leaf_index).
     */
    std::tuple<const std::byte*, std::size_t, uint64_t> get_leaf() const
    {
      return {m_leaf.empty() ? nullptr : m_leaf.data(), m_leaf.size(), m_leaf_index};
    }

    /**
     * @brief Returns a pair containing the pointer to the root data and its size.
     * @return A pair of (root data pointer, root size).
     */
    std::pair<const std::byte*, std::size_t> get_root() const
    {
      return {m_root.empty() ? nullptr : m_root.data(), m_root.size()};
    }

    /**
     * @brief Adds a node to the Merkle path using raw byte data.
     *
     * This function resizes the internal path buffer to accommodate the new node and then copies
     * the provided byte data into the newly allocated space.
     *
     * @param node Pointer to the node data as a sequence of bytes.
     * @param size Size of the node data in bytes.
     */
    void push_node_to_path(const std::byte* node, uint64_t size)
    {
      std::size_t old_size = m_path.size();
      m_path.resize(old_size + size);
      ICICLE_CHECK(icicle_copy(m_path.data() + old_size, node, size));
    }

    /**
     * @brief Adds a node to the Merkle path using a typed object.
     *
     * This templated function accepts any type of node, calculates its size, and forwards the
     * data to the byte-based version of `push_node_to_path()`.
     *
     * @tparam T Type of the node to add to the Merkle path.
     * @param node The node data to add to the Merkle path.
     */
    template <typename T>
    void push_node_to_path(const T& node)
    {
      push_node_to_path(reinterpret_cast<const std::byte*>(&node), sizeof(T));
    }

    /**
     * @brief Pre-allocate the path to a given size and return a pointer to the allocated memory.
     * @param size The size to pre-allocate for the path, in bytes.
     * @return std::byte* Pointer to the allocated memory.
     */
    std::byte* allocate_path_and_get_ptr(std::size_t size)
    {
      m_path.resize(size);  // Resize the path vector to the desired size
      return m_path.data(); // Return a pointer to the beginning of the data
    }

    /**
     * @brief Accesses the Merkle path at a specific byte offset and casts the data to the desired type.
     *
     * This function allows access to the Merkle path at a given byte offset, interpreting the data
     * as a type `T`. It checks if the offset is within bounds before performing the cast. If the
     * offset is out of bounds, an exception is thrown.
     *
     * @tparam T The type to cast the data to (e.g., struct or primitive type).
     * @param offset The byte offset from the beginning of the Merkle path.
     * @return A pointer to the data at the specified offset, cast to the requested type `T`.
     * @throws std::out_of_range If the offset is beyond the size of the path.
     */
    template <typename T>
    const T* access_path_at_offset(uint64_t offset)
    {
      if (offset >= m_path.size()) { throw std::out_of_range("Offset out of bounds"); }
      return reinterpret_cast<const T*>(m_path.data() + offset);
    }

  private:
    bool m_pruned{false};
    uint64_t m_leaf_index{0};
    std::vector<std::byte> m_leaf;
    std::vector<std::byte> m_root;
    std::vector<std::byte> m_path;
  };

} // namespace icicle