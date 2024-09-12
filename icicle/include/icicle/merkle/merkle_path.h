#pragma once

#include <memory>
#include <cstddef>
#include <vector>
#include <iostream> // For streams
#include <stdexcept>

namespace icicle {

  /**
   * @brief Class representing the Merkle path in a move-only manner.
   *
   * This class manages a Merkle path as a collection of bytes. It is designed to be move-only,
   * meaning it can be transferred but not copied, ensuring clear ownership of the path data.
   * The path is stored using a `std::unique_ptr` for exclusive ownership.
   */
  class MerklePath
  {
  public:
    /**
     * @brief Constructor for MerklePath.
     * @param size The size of the path data.
     * @param pruned Whether the Merkle path is pruned.
     */
    explicit MerklePath(uint64_t size, bool pruned = false)
        : m_size(size), m_pruned(pruned), m_path_data(std::make_unique<std::byte[]>(size))
    {
    }

    // Move constructor
    MerklePath(MerklePath&& other) noexcept
        : m_path_data(std::move(other.m_path_data)), m_size(other.m_size), m_pruned(other.m_pruned)
    {
      other.m_size = 0;
      other.m_pruned = false;
    }

    // Move assignment
    MerklePath& operator=(MerklePath&& other) noexcept
    {
      if (this != &other) {
        m_path_data = std::move(other.m_path_data);
        m_size = other.m_size;
        m_pruned = other.m_pruned;
        other.m_size = 0;
        other.m_pruned = false;
      }
      return *this;
    }

    // Deleted copy constructor and assignment operator
    MerklePath(const MerklePath&) = delete;
    MerklePath& operator=(const MerklePath&) = delete;

    /**
     * @brief Get a pointer to the path data.
     * @return Pointer to the path data.
     */
    const std::byte* data() const { return m_path_data.get(); }

    /**
     * @brief Get a pointer to the path data for modification.
     * @return Pointer to the path data.
     */
    std::byte* data() { return m_path_data.get(); }

    /**
     * @brief Get the size of the path data.
     * @return The size of the path data.
     */
    uint64_t size() const { return m_size; }

    /**
     * @brief Check if the Merkle path is pruned.
     * @return True if the path is pruned, false otherwise.
     */
    bool is_pruned() const { return m_pruned; }

    /**
     * @brief Serialize the Merkle path into an output stream.
     * @param os The output stream to serialize into.
     */
    void serialize(std::ostream& os) const { os.write(reinterpret_cast<const char*>(m_path_data.get()), m_size); }

    /**
     * @brief Deserialize data from an input stream.
     * @param is The input stream to deserialize from.
     */
    void deserialize(std::istream& is)
    {
      is.read(reinterpret_cast<char*>(m_path_data.get()), m_size);
      if (!is) { throw std::runtime_error("Error reading from input stream. Data may be incomplete."); }
    }

    /**
     * @brief Access data at a specific offset and cast it to the desired type.
     * @tparam T The type to cast the data to.
     * @param offset The byte offset to access.
     * @return Pointer to the data cast to type T.
     */
    template <typename T>
    T* access_at_offset(uint64_t offset)
    {
      if (offset >= m_size) { throw std::out_of_range("Offset out of bounds"); }
      return reinterpret_cast<T*>(m_path_data.get() + offset);
    }

  private:
    std::unique_ptr<std::byte[]> m_path_data; ///< Unique ownership of the path data.
    uint64_t m_size;                          ///< Size of the path data.
    bool m_pruned;                            ///< Whether the Merkle path is pruned.
  };

} // namespace icicle