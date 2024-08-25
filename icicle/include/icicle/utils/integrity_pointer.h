#pragma once

#include <memory>
#include <iostream>
#include <stdexcept>

/**
 * @brief A template class that wraps a raw pointer with additional checks for data integrity.
 *
 * IntegrityPointer is designed to wrap a raw pointer and associate it with a validation
 * mechanism based on a counter. This counter is monitored via a std::weak_ptr, allowing
 * the IntegrityPointer to check if the data it points to has potentially been invalidated.
 * It is intended for scenarios where there's a need to ensure the integrity of the pointed-to
 * data throughout the lifetime of the pointer, particularly useful in complex systems where
 * data validity can change over time due to external factors.
 *
 * Usage involves providing the raw pointer to be wrapped, a std::weak_ptr to a counter, and
 * the expected value of that counter. The IntegrityPointer can then be used much like a normal
 * pointer, with the addition of integrity checks before access.
 *
 * @tparam T The type of the pointed-to object.
 */

template <typename T>
class IntegrityPointer
{
public:
  /**
   * Constructs an IntegrityPointer wrapping a raw pointer with a validity check based on a counter.
   *
   * @param ptr A raw pointer to the data of type T.
   * @param counterWeakPtr A std::weak_ptr to an int counter, used for validation.
   * @param expectedCounterValue The expected value of the counter for the pointer to be considered valid.
   */
  IntegrityPointer(const T* ptr, std::weak_ptr<int> counterWeakPtr, int expectedCounterValue)
      : m_ptr(ptr), m_counterWeakPtr(counterWeakPtr), m_expectedCounterValue(expectedCounterValue)
  {
  }
  IntegrityPointer(const IntegrityPointer& other) = default;
  IntegrityPointer(IntegrityPointer&& other) = default;

  /**
   * Retrieves the raw pointer. Use with caution, as direct access bypasses validity checks.
   * @return T* The raw pointer to the data.
   */
  const T* get() const { return isValid() ? m_ptr : nullptr; }

  /**
   * Dereferences the pointer. Throws std::runtime_error if the pointer is invalid.
   * @return A reference to the data pointed to by the raw pointer.
   */
  const T& operator*() const
  {
    assertValid();
    return *m_ptr;
  }

  /**
   * Provides access to the member of the pointed-to object. Throws std::runtime_error if the pointer is invalid.
   * @return T* The raw pointer to the data.
   */
  const T* operator->() const
  {
    assertValid();
    return m_ptr;
  }

  /**
   * Checks whether the pointer is still considered valid by comparing the current value of the counter
   * to the expected value.
   * @return true if the pointer is valid, false otherwise.
   */
  bool isValid() const
  {
    if (auto counterSharedPtr = m_counterWeakPtr.lock()) { return *counterSharedPtr == m_expectedCounterValue; }
    return false;
  }

private:
  const T* m_ptr;                      ///< The raw pointer to the data.
  std::weak_ptr<int> m_counterWeakPtr; ///< A weak pointer to the counter used for validation.
  const int m_expectedCounterValue;    ///< The expected value of the counter for the pointer to be valid.

  /**
   * Asserts the validity of the pointer. Throws std::runtime_error if the pointer is invalid.
   */
  void assertValid() const
  {
    if (!isValid()) {
      logInvalidAccess();
      throw std::runtime_error("Attempted to access invalidated IntegrityPointer.");
    }
  }

  /**
   * Logs an attempt to access an invalidated pointer.
   */
  static void logInvalidAccess()
  {
    std::cerr << "Warning: Attempted to access invalidated IntegrityPointer." << std::endl;
  }
};
