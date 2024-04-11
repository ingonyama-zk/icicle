
#include "polynomials/tracing/graph_visualizer_visitor.h"

namespace polynomials {

  template <typename C, typename D, typename I>
  void GraphvizVisualizer<C, D, I>::visit(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node)
  {
    if (this->visited(node, true /*=set_visited*/)) return;

    auto id = node->m_id;
    auto memid = node->m_memory_context->m_id;

    m_out_stream << node->m_id << " [label=\"" << OpcodeToStr(node->m_opcode) << " (id=" << node->m_id
                 << ", memid=" << memid << ")\n"
                 << node->m_attrs.to_string() << "\"];\n";
    for (auto op : node->get_operands()) {
      visit(op);
      m_out_stream << op->m_id << " -> " << node->m_id << "\n";
    }
  }

  template <typename C, typename D, typename I>
  void GraphvizVisualizer<C, D, I>::run(Polynomial<C, D, I>& p)
  {
    auto trace_ctxt = dynamic_cast<TracingPolynomialContext<C, D, I>*>(p.get_context());
    if (!trace_ctxt) {
      std::cerr << "[WARNING] Graph visualizer expecting TracingPolynomialContext. draw skipped.\n";
      return;
    }

    auto trace_ctxt_shared = trace_ctxt->getptr();
    run(trace_ctxt_shared);
  }

  template <typename C, typename D, typename I>
  void GraphvizVisualizer<C, D, I>::run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context)
  {
    m_out_stream.clear();
    this->m_visited.clear();
    m_out_stream << "digraph G {\n";
    visit(context);
    m_out_stream << "}";
  }

  template class GraphvizVisualizer<>;

} // namespace polynomials