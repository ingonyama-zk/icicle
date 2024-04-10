#pragma once

#include <deque>

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"

namespace polynomials {

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class Interpreter
  {
  private:
    std::shared_ptr<IPolynomialBackend<C, D, I>> m_compute_backend;
    std::set<uint64_t> m_visited;
    std::deque<std::shared_ptr<TracingPolynomialContext<C, D, I>>> m_bfs_order;

    void evaluate(std::shared_ptr<TracingPolynomialContext<C, D, I>> node);
    void evaluate_single_node(std::shared_ptr<TracingPolynomialContext<C, D, I>> node);
    bool visited(std::shared_ptr<TracingPolynomialContext<C, D, I>> node, bool set_visited);

  public:
    Interpreter(std::shared_ptr<IPolynomialBackend<C, D, I>> compute_backend) : m_compute_backend(compute_backend) {}
    void run(std::shared_ptr<TracingPolynomialContext<C, D, I>> context);
  };

  extern template class Interpreter<>;

} // namespace polynomials