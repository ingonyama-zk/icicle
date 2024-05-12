#pragma once

#include <utility>   // for std::pair
#include <tuple>     // for std::tuple
#include <iostream>  // for std::ostream
#include <algorithm> // for std::max
#include <cstdint>   // for uint64_t, etc.
#include <memory>
#include "../utils/integrity_pointer.h"

namespace polynomials {

  template <typename Coeff, typename Domain, typename Image>
  class IPolynomialBackend;

  /**
   * @brief Interface for polynomial context, encapsulating state, memory, and device context.
   *
   * This interface is designed to manage the state of polynomials including their coefficients and
   * evaluations in both natural and reversed order. It supports operations for converting between
   * these forms, allocating and releasing resources, and accessing the underlying data. The context
   * abstracts over the specifics of memory and execution context, allowing polynomials to be managed
   * in a way that is agnostic to the underlying hardware or software infrastructure.
   *
   * @tparam C Type of the coefficients.
   * @tparam D Domain type, representing the input space of the polynomial.
   * @tparam I Image type, representing the output space of the polynomial.
   */
  template <typename C, typename D, typename I>
  class IPolynomialContext
  {
  public:
    friend class IPolynomialBackend<C, D, I>;

    // Enumerates the possible states of a polynomial context.
    enum State { Invalid, Coefficients, EvaluationsOnRou_Natural, EvaluationsOnRou_Reversed };

    // The size of the largest element among coefficients and evaluations.
    static constexpr size_t ElementSize = std::max(sizeof(C), sizeof(I));

    /**
     * @brief Construct a new IPolynomialContext object.
     */
    IPolynomialContext() : m_id{s_id++} {}

    /**
     * @brief Virtual destructor for IPolynomialContext.
     */
    virtual ~IPolynomialContext() = default;

    // Methods for initializing the context from coefficients or evaluations.
    virtual void from_coefficients(uint64_t nof_coefficients, const C* coefficients = nullptr) = 0;
    virtual void from_rou_evaluations(uint64_t nof_evaluations, const I* evaluations = nullptr) = 0;

    // Method for cloning the context from another instance.
    virtual void clone(IPolynomialContext& from) = 0;

    // Methods for resource management.
    virtual void allocate(uint64_t nof_elements, State init_state = State::Coefficients, bool memset_zeros = true) = 0;
    virtual void release() = 0;

    // Methods for transforming between coefficients and evaluations.
    virtual void transform_to_coefficients(uint64_t nof_coefficients = 0) = 0;
    virtual void transform_to_evaluations(uint64_t nof_evaluations = 0, bool is_reversed = false) = 0;

    // Accessors for the state and number of elements.
    virtual State get_state() const = 0;
    virtual uint64_t get_nof_elements() const = 0;

    // Methods to get direct access to coefficients and evaluations.
    virtual std::pair<const C*, uint64_t> get_coefficients() = 0;
    virtual std::pair<const I*, uint64_t> get_rou_evaluations() = 0;

    // Methods to get views of coefficients and evaluations, including device id.
    virtual std::tuple<IntegrityPointer<C>, uint64_t /*size*/, uint64_t /*device_id*/> get_coefficients_view() = 0;
    virtual std::tuple<IntegrityPointer<I>, uint64_t /*size*/, uint64_t /*device_id*/>
    get_rou_evaluations_view(uint64_t nof_evaluations = 0, bool is_reversed = false) = 0;

    // Method for printing the context state to an output stream.
    virtual void print(std::ostream& os) = 0;

  protected:
    // Provides mutable access to the underlying storage for backend computations.
    virtual void* get_storage_mutable() = 0;
    virtual const void* get_storage_immutable() = 0;

    // Static and instance variables for debug id management.
    static inline uint64_t s_id = 0; // Global id counter.

  public:
    const uint64_t m_id;
  };
} // namespace polynomials
