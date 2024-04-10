#pragma once

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"
#include <list>
#include <ostream>
#include <set>

namespace polynomials {

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class GraphvizVisualizer
  {
  private:
    std::ostream& m_out_stream;
    std::set<uint64_t> m_visited;

    void visit(TracingPolynomialContext<C, D, I>* context);

  public:
    GraphvizVisualizer(std::ostream& stream) : m_out_stream{stream} {}
    void run(Polynomial<C, D, I>& p);
  };

  extern template class GraphvizVisualizer<>;

} // namespace polynomials