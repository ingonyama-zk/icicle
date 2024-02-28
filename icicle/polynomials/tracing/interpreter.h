#pragma once

#include "curves/curve_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"

namespace polynomials {

  template <typename C = curve_config::scalar_t, typename D = C, typename I = C>
  class Interpreter
  {
  private:
    std::shared_ptr<IPolynomialBackend<C, D, I>> m_compute_backend;
    void visit(std::shared_ptr<TracingPolynomialContext<C, D, I>> context);

  public:
    Interpreter(std::shared_ptr<IPolynomialBackend<C, D, I>> compute_backend) : m_compute_backend(compute_backend) {}
    void run(std::shared_ptr<TracingPolynomialContext<C, D, I>> context);
  };

  extern template class Interpreter<>;

} // namespace polynomials