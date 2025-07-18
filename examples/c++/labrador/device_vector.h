#pragma once

#include "icicle/runtime.h" // icicle_malloc, icicle_free, …
#include "icicle/errors.h"  // ICICLE_CHECK

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @class deviceVector
 * @brief A dynamic array container for GPU/device memory
 *
 * This class provides an STL vector-like interface for managing memory on
 * compute devices (GPU, etc). Data is stored on the device and must be
 * explicitly copied to/from host memory.
 *
 * @tparam T The type of elements stored in the vector
 */
template <typename T>
class deviceVector
{
private:
  T* data_;         ///< Pointer to device memory
  size_t size_;     ///< Number of elements currently stored
  size_t capacity_; ///< Allocated capacity in number of elements

  /**
   * @brief Grows the vector capacity
   * @param new_capacity Minimum capacity needed
   *
   * Allocates new device memory with at least new_capacity elements.
   * Typically doubles the current capacity if new_capacity is only
   * slightly larger than current capacity.
   */
  void grow(size_t new_capacity);

public:
  /**
   * @brief Default constructor
   * Creates an empty vector with no allocated memory
   */
  deviceVector() : data_(nullptr), size_(0), capacity_(0) {}

  /**
   * @brief Size constructor
   * @param size Number of elements to allocate
   *
   * Creates a vector with the specified size. Elements are
   * uninitialized on the device.
   */
  explicit deviceVector(size_t size);

  /**
   * @brief Constructor from std::vector
   * @param other Source vector to copy from host memory
   *
   * Allocates device memory and copies all elements from the host vector
   */
  explicit deviceVector(const std::vector<T>& other);

  /**
   * @brief Copy constructor
   * @param other Source deviceVector to copy
   *
   * Creates a deep copy with new device memory allocation
   */
  deviceVector(const deviceVector& other);

  /**
   * @brief Destructor
   * Frees all allocated device memory
   */
  ~deviceVector();

  /**
   * @brief Assignment operator
   * @param other Source deviceVector to copy
   * @return Reference to this vector
   *
   * Performs deep copy, reusing existing allocation if possible
   */
  deviceVector& operator=(const deviceVector& other);

  // Capacity operations
  /**
   * @brief Get the number of elements
   * @return Current number of elements in the vector
   */
  size_t size() const { return size_; }

  /**
   * @brief Get the allocated capacity
   * @return Number of elements that can be stored without reallocation
   */
  size_t capacity() const { return capacity_; }

  /**
   * @brief Check if vector is empty
   * @return true if size is 0, false otherwise
   */
  bool empty() const { return size_ == 0; }

  /**
   * @brief Reserve capacity
   * @param new_capacity Minimum capacity to reserve
   *
   * Ensures the vector can hold at least new_capacity elements
   * without reallocation. Does nothing if current capacity is sufficient.
   */
  void reserve(size_t new_capacity);

  /**
   * @brief Resize the vector
   * @param new_size New number of elements
   *
   * Changes the size of the vector. If new_size > size, new elements
   * are uninitialized. If new_size < size, elements are truncated.
   */
  void resize(size_t new_size);

  /**
   * @brief Clear all elements
   *
   * Sets size to 0 but keeps allocated memory
   */
  void clear() { size_ = 0; }

  // Element insertion
  /**
   * @brief Append a single element from host memory
   * @param value Element to copy to device and append
   */
  void push_back(const T& value);

  /**
   * @brief Append elements from another deviceVector
   * @param vec Source vector on device
   *
   * Performs device-to-device copy
   */
  void push_back(const deviceVector<T>& vec);

  /**
   * @brief Append elements from std::vector
   * @param vec Source vector in host memory
   *
   * Performs host-to-device copy
   */
  void push_back(const std::vector<T>& vec);

  // Data transfer operations
  /**
   * @brief Replace contents with data from host memory
   * @param host_data Pointer to host memory
   * @param count Number of elements to copy
   *
   * Resizes the vector to count and copies all data from host
   */
  void copy_from_host(const T* host_data, size_t count);

  /**
   * @brief Copy elements to host memory
   * @param host_data Pointer to host memory buffer
   * @param count Maximum number of elements to copy
   *
   * Copies min(count, size()) elements to the host buffer
   */
  void copy_to_host(T* host_data, size_t count) const;

  /**
   * @brief Convert to std::vector
   * @return A new std::vector containing all elements
   *
   * Allocates a new std::vector and copies all elements from device
   */
  std::vector<T> as_host_vector() const;

  // Element access
  /**
   * @brief Get a single element
   * @param idx Index of element to retrieve
   * @return Copy of the element at index idx
   *
   * Copies a single element from device to host
   */
  T get(size_t idx) const;

  /**
   * @brief Set a single element
   * @param idx Index of element to set
   * @param value Value to copy to device
   *
   * Copies a single element from host to device
   */
  void set(size_t idx, const T& value);

  /**
   * @brief Get a slice as std::vector
   * @param start Starting index (inclusive)
   * @param stop Ending index (exclusive)
   * @return std::vector containing elements [start, stop)
   *
   * Returns elements in range [start, stop) similar to Python slicing.
   * Bounds are automatically clamped to valid range.
   */
  std::vector<T> slice(size_t start, size_t stop) const;

  /**
   * @brief Get raw device pointer
   * @return Pointer to device memory
   *
   * @warning This returns a device pointer that cannot be dereferenced
   * in host code. Use only for passing to device kernels or device operations.
   */
  T* data() { return data_; }

  /**
   * @brief Get raw device pointer (const version)
   * @return Const pointer to device memory
   *
   * @warning This returns a device pointer that cannot be dereferenced
   * in host code. Use only for passing to device kernels or device operations.
   */
  const T* data() const { return data_; }
};

// Definitions

template <typename T>
deviceVector<T>::deviceVector() : data_(nullptr), size_(0), capacity_(0)
{
}

template <typename T>
deviceVector<T>::~deviceVector()
{
  if (data_) { ICICLE_CHECK(icicle_free(data_)); }
}

// Private helper - grow function
// Growth strategy: double the capacity or use min_capacity, whichever is larger
template <typename T>
void deviceVector<T>::grow(size_t min_capacity)
{
  if (min_capacity <= capacity_) { return; }

  // Growth strategy: double the capacity or use min_capacity, whichever is larger
  size_t new_capacity = capacity_ == 0 ? 1 : capacity_ * 2;
  if (new_capacity < min_capacity) { new_capacity = min_capacity; }

  // Allocate new memory
  T* new_data = nullptr;
  ICICLE_CHECK(icicle_malloc(reinterpret_cast<void**>(&new_data), new_capacity * sizeof(T)));

  // Copy existing data if any
  if (data_ && size_ > 0) { ICICLE_CHECK(icicle_copy(new_data, data_, size_ * sizeof(T))); }

  // Free old memory
  if (data_) { ICICLE_CHECK(icicle_free(data_)); }

  data_ = new_data;
  capacity_ = new_capacity;
}

// Reserve function
template <typename T>
void deviceVector<T>::reserve(size_t new_capacity)
{
  if (new_capacity > capacity_) { grow(new_capacity); }
}

// Size constructor: Note it doesn't set the memory to the default constructor-- it sets all bytes to 0 in the memory
template <typename T>
deviceVector<T>::deviceVector(size_t count) : data_(nullptr), size_(count), capacity_(count)
{
  if (count > 0) {
    ICICLE_CHECK(icicle_malloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    ICICLE_CHECK(icicle_memset(data_, 0, count * sizeof(T)));
  }
}

// Doesn't reduce the allocated memory
template <typename T>
void deviceVector<T>::clear()
{
  size_ = 0;
  // Note: we keep the capacity and allocated memory
}

// Resize function
template <typename T>
void deviceVector<T>::resize(size_t new_size)
{
  if (new_size > capacity_) { grow(new_size); }

  // If growing, zero-initialize new elements
  if (new_size > size_) {
    size_t bytes_to_zero = (new_size - size_) * sizeof(T);
    ICICLE_CHECK(icicle_memset(data_ + size_, 0, bytes_to_zero));
  }

  size_ = new_size;
}

// 4. Push_back function
template <typename T>
void deviceVector<T>::push_back(const T& value)
{
  if (size_ == capacity_) { grow(size_ + 1); }

  // Copy single element from host to device
  ICICLE_CHECK(icicle_copy(data_ + size_, &value, sizeof(T)));
  size_++;
}

// Push_back for deviceVector
template <typename T>
void deviceVector<T>::push_back(const deviceVector<T>& vec)
{
  if (vec.size_ == 0) { return; }

  size_t new_size = size_ + vec.size_;
  if (new_size > capacity_) { grow(new_size); }

  // Copy device to device
  ICICLE_CHECK(icicle_copy(data_ + size_, vec.data_, vec.size_ * sizeof(T)));
  size_ = new_size;
}

// Push_back for std::vector
template <typename T>
void deviceVector<T>::push_back(const std::vector<T>& vec)
{
  if (vec.size() == 0) { return; }

  size_t new_size = size_ + vec.size();
  if (new_size > capacity_) { grow(new_size); }

  // Copy host to device
  ICICLE_CHECK(icicle_copy_to_device(data_ + size_, vec.data(), vec.size() * sizeof(T)));
  size_ = new_size;
}

// Copy from host function
template <typename T>
void deviceVector<T>::copy_from_host(const T* host_data, size_t count)
{
  if (count == 0) { return; }

  // Resize if necessary
  if (count > capacity_) {
    if (data_) { ICICLE_CHECK(icicle_free(data_)); }
    ICICLE_CHECK(icicle_malloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    capacity_ = count;
  }

  // Copy data from host to device
  ICICLE_CHECK(icicle_copy_to_device(data_, host_data, count * sizeof(T)));
  size_ = count;
}

// Constructor from std::vector
template <typename T>
deviceVector<T>::deviceVector(const std::vector<T>& other) : data_(nullptr), size_(0), capacity_(0)
{
  copy_from_host(other.data(), other.size());
}

// Copy to host function
template <typename T>
void deviceVector<T>::copy_to_host(T* host_data, size_t count) const
{
  if (count == 0 || size_ == 0) { return; }

  // Copy min(count, size_) elements
  size_t elements_to_copy = count < size_ ? count : size_;
  icicle_copy_to_host(host_data, data_, elements_to_copy * sizeof(T));
}
template <typename T>
std::vector<T> deviceVector<T>::as_host_vector() const
{
  std::vector<T> v(size_);
  copy_to_host(v.data(), size_);
  return v;
}

// Copy constructor
template <typename T>
deviceVector<T>::deviceVector(const deviceVector& other) : data_(nullptr), size_(0), capacity_(0)
{
  if (other.size_ > 0) {
    ICICLE_CHECK(icicle_malloc(reinterpret_cast<void**>(&data_), other.size_ * sizeof(T)));
    size_ = other.size_;
    capacity_ = other.size_;

    // Device to device copy
    ICICLE_CHECK(icicle_copy(data_, other.data_, size_ * sizeof(T)));
  }
}

// Assignment operator
template <typename T>
deviceVector<T>& deviceVector<T>::operator=(const deviceVector& other)
{
  if (this != &other) {
    // Resize if necessary
    if (other.size_ > capacity_) {
      if (data_) { ICICLE_CHECK(icicle_free(data_)); }
      ICICLE_CHECK(icicle_malloc(reinterpret_cast<void**>(&data_), other.size_ * sizeof(T)));
      capacity_ = other.size_;
    }

    size_ = other.size_;

    // Copy data
    if (size_ > 0) { ICICLE_CHECK(icicle_copy(data_, other.data_, size_ * sizeof(T))); }
  }
  return *this;
}

// Get single element (copies from device to host)
template <typename T>
T deviceVector<T>::get(size_t idx) const
{
  if (idx >= size_) { throw std::out_of_range("deviceVector::get: index out of range"); }
  T value;
  ICICLE_CHECK(icicle_copy_to_host(&value, data_ + idx, sizeof(T)));
  return value;
}

// Set single element (copies from host to device)
template <typename T>
void deviceVector<T>::set(size_t idx, const T& value)
{
  if (idx >= size_) { throw std::out_of_range("deviceVector::set: index out of range"); }
  ICICLE_CHECK(icicle_copy_to_device(data_ + idx, &value, sizeof(T)));
}

template <typename T>
std::vector<T> deviceVector<T>::slice(size_t start, size_t stop) const
{
  // Handle bounds
  if (start >= size_ || stop > size_) { throw std::out_of_range("deviceVector::slice: index out of range"); }
  if (start > stop || stop > size_) { throw std::out_of_range("deviceVector::slice: start > stop"); }

  size_t count = stop - start;
  std::vector<T> result(count);

  // Copy slice from device to host
  icicle_copy_to_host(result.data(), data_ + start, count * sizeof(T));

  return result;
}
