#pragma once

#include "polynomial_context.h"
#include "polynomial_backend.h"
#include <memory> // For std::shared_ptr

namespace icicle {

  /**
   * @brief Abstract factory for creating polynomial contexts and backends.
   *
   * The `AbstractPolynomialFactory` serves as an interface for factories capable of creating
   * instances of `IPolynomialContext` and `IPolynomialBackend`. This design allows for the
   * decoupling of object creation from their usage, facilitating the implementation of various
   * computational strategies (e.g., GPU, ZPU) without altering client code. Each concrete factory
   * is expected to provide tailored implementations of polynomial contexts and backends that
   * are optimized for specific computational environments.
   *
   * @tparam C Type of the coefficients.
   * @tparam D Domain type, representing the input space of the polynomial.
   * @tparam I Image type, representing the output space of the polynomial.
   */
  template <typename C, typename D = C, typename I = C>
  class AbstractPolynomialFactory
  {
  public:
    /**
     * @brief Creates and returns a shared pointer to an `IPolynomialContext` instance.
     *
     * @return std::shared_ptr<IPolynomialContext<C, D, I>> A shared pointer to the created
     *         polynomial context instance.
     */
    virtual std::shared_ptr<IPolynomialContext<C, D, I>> create_context() = 0;

    /**
     * @brief Creates and returns a shared pointer to an `IPolynomialBackend` instance.
     *
     * @return std::shared_ptr<IPolynomialBackend<C, D, I>> A shared pointer to the created
     *         polynomial backend instance.
     */
    virtual std::shared_ptr<IPolynomialBackend<C, D, I>> create_backend() = 0;

    /**
     * @brief Virtual destructor for the `AbstractPolynomialFactory`.
     */
    virtual ~AbstractPolynomialFactory() = default;
  };

} // namespace icicle
