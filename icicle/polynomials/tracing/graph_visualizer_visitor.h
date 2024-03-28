#pragma once

#include "curves/curve_config.cuh"
#include "polynomials/polynomials.h"
#include <ostream>

namespace polynomials {

  template <typename C = curve_config::scalar_t, typename D = C, typename I = C>
  class GraphvizVisualizer : public IContextVisitor<C, D, I>
  {
  private:
    std::ostream& m_out_stream;

    void visit(IPolynomialContext<C, D, I>* context)
    {
      // TODO Yuval: add visited flag ??
      m_out_stream << context->m_id << " [label=\"" << OpcodeToStr(context->m_op.opcode) << " (id=" << context->m_id
                   << ")\"];\n";
      for (auto& arg : context->m_args) {
        arg->accept(this);
        m_out_stream << arg->m_id << " -> " << context->m_id << "\n";
      }
    }

  public:
    GraphvizVisualizer(std::ostream& stream) : m_out_stream{stream} {}
    void run(Polynomial<C, D, I>& p)
    {
      m_out_stream.clear();
      m_out_stream << "digraph G {\n";
      p.get_context()->accept(this);
      m_out_stream << "}";
    }
  };

} // namespace polynomials