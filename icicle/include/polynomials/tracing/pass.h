#pragma once

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"

namespace polynomials {

  // Pass is a base class for all passes. It defines an interface for a pass manager to use
  // TODO Yuval: add pass deps so that a pass manager can sort based on that and call passes in a valid order and
  // identify loops

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class Pass
  {
  protected:
    std::set<uint64_t> m_visited;

    bool visited(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node, bool set_visited)
    {
      const bool is_visited = m_visited.find(node->m_id) != m_visited.end();
      if (set_visited) m_visited.insert(node->m_id);
      return is_visited;
    }

  public:
    Pass() = default;
    virtual void run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context) = 0;
  };
} // namespace polynomials