#include "polynomials/tracing/polynomial_ops.h"
#include "polynomials/tracing/memory_management.h"

namespace polynomials {

  template <typename C, typename D, typename I>
  bool
  MemoryManagement<C, D, I>::is_compatible(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node, uint64_t min_size)
  {
    // cannot use memory of a bound operand since the polynomial it is bound to would be wrong
    if (node->is_bound()) return false;
    // TODO consider the size
    for (auto& weak_dep : node->get_dependents()) {
      if (auto shared_dep = weak_dep.lock()) {
        // if already sharing memory with another dep, then cannot reuse
        if (shared_dep->m_memory_context == node->m_memory_context) return false;
      }
    }

    return true;
  }

  template <typename C, typename D, typename I>
  void MemoryManagement<C, D, I>::visit(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node)
  {
    if (this->visited(node, true /*=set_visited*/) || node->is_evaluated()) return;

    for (auto op : node->get_operands()) {
      visit(op);
    }

    // The idea is that I can reuse and operand's memory context
    // compatible operand is one that is unbound (therefore no polynomial owns this memory)

    switch (node->m_opcode) {
    case eOpcode::SCALAR_MUL:
    case eOpcode::CLONE: {
      // take operand memory if compatible
      auto op = node->get_operand(0);
      if (is_compatible(op)) { node->m_memory_context = op->m_memory_context; }
    } break;
    case eOpcode::ADD:
    case eOpcode::SUB: {
      auto op0 = node->get_operand(0);
      auto op1 = node->get_operand(1);
      if (is_compatible(op0)) {
        node->m_memory_context = op0->m_memory_context;
      } else if (is_compatible(op1)) {
        node->m_memory_context = op1->m_memory_context;
      }
    } break;
    case eOpcode::DIV_BY_VANISHING: {
      // TODO Yuval: can do this if the computation is
    } break;
    case eOpcode::MUL: {
      // Check if the division can be done inplace and have a compatible operand
      // TODO Yuval
    } break;
    default:
      break;
    }
  }

  template <typename C, typename D, typename I>
  void MemoryManagement<C, D, I>::run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node)
  {
    this->m_visited.clear();
    visit(node);
  }

  template class MemoryManagement<>;
} // namespace polynomials
