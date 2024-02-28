
#include "polynomials/tracing/graph_visualizer_visitor.h"

namespace polynomials {

  template <typename C, typename D, typename I>
  void GraphvizVisualizer<C, D, I>::visit(std::shared_ptr<TracingPolynomialContext<C, D, I>> context)
  {
    if (m_visited.find(context->m_id) != m_visited.end()) return;
    m_visited.insert(context->m_id);

    auto id = context->m_id;
    auto memid = context->m_memory_context->m_id;

    m_out_stream << context->m_id << " [label=\"" << OpcodeToStr(context->m_opcode) << " (id=" << context->m_id
                 << ", memid=" << memid << ")\n"
                 << context->m_attrs.to_string() << "\"];\n";
    for (auto& op : context->get_operands()) {
      visit(op);
      m_out_stream << op->m_id << " -> " << context->m_id << "\n";
    }
  }

  template <typename C, typename D, typename I>
  void GraphvizVisualizer<C, D, I>::run(Polynomial<C, D, I>& p)
  {
    auto trace_ctxt = std::dynamic_pointer_cast<TracingPolynomialContext<C, D, I>>(p.get_context());
    if (!trace_ctxt) {
      std::cerr << "[WARNING] Graph visualizer expecting TracingPolynomialContext. draw skipped.\n";
      return;
    }

    m_out_stream.clear();
    m_visited.clear();
    m_out_stream << "digraph G {\n";
    visit(trace_ctxt);
    m_out_stream << "}";
  }

  template class GraphvizVisualizer<>;

} // namespace polynomials