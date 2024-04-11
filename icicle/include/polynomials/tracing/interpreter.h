#pragma once

#include <deque>

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"
#include "polynomials/tracing/pass.h"

namespace polynomials {

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class Interpreter : public Pass<C, D, I>
  {
  private:
    std::shared_ptr<IPolynomialBackend<C, D, I>> m_compute_backend;
    std::deque<std::shared_ptr<TracingPolynomialContext<C, D, I>>> m_bfs_order;

    void evaluate(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node);
    void evaluate_single_node(std::shared_ptr<TracingPolynomialContext<C, D, I>> node);

  public:
    Interpreter(std::shared_ptr<IPolynomialBackend<C, D, I>> compute_backend) : m_compute_backend(compute_backend) {}
    void run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context) override;
  };

  extern template class Interpreter<>;

} // namespace polynomials