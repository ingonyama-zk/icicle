#include "polynomials/tracing/polynomial_ops.h"
#include "polynomials/tracing/fuse_mac.h"
#include "fields/field_config.cuh"

namespace polynomials {

  template <typename C, typename D, typename I>
  void FuseMac<C, D, I>::visit(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node)
  {
    if (this->visited(node, true) || node->is_evaluated()) return;

    for (auto op : node->get_operands()) {
      visit(op);
    }

    const bool is_add_sub = node->m_opcode == eOpcode::ADD; //|| node->m_opcode == eOpcode::SUB;
    if (!is_add_sub) return;

    auto op_a = node->get_operand(0);
    auto op_b = node->get_operand(1);
    const bool is_op_a_scalar_mul = op_a->m_opcode == eOpcode::SCALAR_MUL;
    const bool is_op_b_scalar_mul = op_b->m_opcode == eOpcode::SCALAR_MUL;

    // TODO Yuval: handle subtraction too

    if (is_op_b_scalar_mul) {
      Attributes& attrs = op_b->m_attrs;
      const auto& scalar = attrs.getAttribute<D>(OP_ATTR_SCALAR);
      node->m_opcode = eOpcode::MAC;
      node->set_operands({op_a, op_b->get_operand(0)});
      node->m_attrs.setAttribute(OP_ATTR_SCALAR, scalar);
    } else if (is_op_a_scalar_mul) {
      Attributes& attrs = op_a->m_attrs;
      const auto& scalar = attrs.getAttribute<D>(OP_ATTR_SCALAR);
      node->m_opcode = eOpcode::MAC;
      node->set_operands({op_b, op_a->get_operand(0)});
      node->m_attrs.setAttribute(OP_ATTR_SCALAR, scalar);
    }
  }

  template <typename C, typename D, typename I>
  void FuseMac<C, D, I>::run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node)
  {
    this->m_visited.clear();
    visit(node);
  }

  template class FuseMac<>;
} // namespace polynomials
