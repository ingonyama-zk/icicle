#pragma once

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"

// TODO Yuval: maybe better do it as part of the Interpreter and not a pass

namespace polynomials {

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class MemoryManagement
  {
  private:
    void visit(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context);

    std::set<uint64_t> m_visited;
    bool visited(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node, bool set_visited);
    bool is_compatible(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node, uint64_t min_size = 0);

  public:
    MemoryManagement() = default;
    void run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context);
  };

  extern template class MemoryManagement<>;

} // namespace polynomials