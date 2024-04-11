#pragma once

#include "fields/field_config.cuh"
#include "polynomials/polynomials.h"
#include "polynomials/tracing/polynomial_tracing_backend.cuh"
#include "polynomials/tracing/pass.h"
#include <list>
#include <ostream>
#include <set>

namespace polynomials {

  template <typename C = field_config::scalar_t, typename D = C, typename I = C>
  class GraphvizVisualizer : public Pass<C, D, I>
  {
  private:
    std::ostream& m_out_stream;
    void visit(std::shared_ptr<TracingPolynomialContext<C, D, I>>&);

  public:
    GraphvizVisualizer(std::ostream& stream) : m_out_stream{stream} {}
    void run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context) override;
    void run(Polynomial<C, D, I>& p);
  };

  extern template class GraphvizVisualizer<>;

} // namespace polynomials