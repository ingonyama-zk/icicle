
#include "polynomials/tracing/interpreter.h"
#include "polynomials/tracing/polynomial_ops.h"
#include <cstdint> // for uint64_t, etc.

namespace polynomials {

  template <typename C, typename D, typename I>
  void Interpreter<C, D, I>::evaluate(std::shared_ptr<TracingPolynomialContext<C, D, I>>& node)
  {
    const bool is_visited = this->visited(node, true /*=set_visited*/);
    if (is_visited || node->is_evaluated()) return;

    std::shared_ptr<TracingPolynomialContext<C, D, I>> inplace_operand = nullptr;
    for (auto op : node->get_operands()) {
      evaluate(op);
      // if node is computed inplace, reusing parent memory, must compute siblings first
      const bool is_inplace_parent = op->m_memory_context == node->m_memory_context;

      if (!is_inplace_parent) continue;
      if (inplace_operand && op != inplace_operand) {
        THROW_ICICLE_ERR(IcicleError_t::UndefinedError, "Cannot have multiple inplace parents");
      }
      inplace_operand = op;
    }

    // visit siblings that share my inplace father to compute them before me
    if (inplace_operand) {
      auto siblings = inplace_operand->get_dependents();
      for (auto& weak_sibling : siblings) {
        if (auto shared_sibling = weak_sibling.lock()) evaluate(shared_sibling);
      }
    }
    evaluate_single_node(node);
  }

  template <typename C, typename D, typename I>
  void Interpreter<C, D, I>::evaluate_single_node(std::shared_ptr<TracingPolynomialContext<C, D, I>> context)
  {
    if (context->is_evaluated()) return;

    Attributes& attrs = context->m_attrs;

    // Note: can pass the context directly rather than the memory context but some backends assume they know the
    // context and cast it (e.g. CUDA backend want to check device-id)

    switch (context->m_opcode) {
    case eOpcode::CLONE:
      m_compute_backend->clone(context->m_memory_context, context->get_op_mem_ctxt(0));
      break;
    case eOpcode::ADD:
      m_compute_backend->add(context->m_memory_context, context->get_op_mem_ctxt(0), context->get_op_mem_ctxt(1));
      break;
    case eOpcode::SUB:
      m_compute_backend->subtract(context->m_memory_context, context->get_op_mem_ctxt(0), context->get_op_mem_ctxt(1));
      break;
    case eOpcode::MUL:
      m_compute_backend->multiply(context->m_memory_context, context->get_op_mem_ctxt(0), context->get_op_mem_ctxt(1));
      break;
    case eOpcode::SCALAR_MUL: {
      const auto& scalar = attrs.getAttribute<D>(OP_ATTR_SCALAR);
      m_compute_backend->multiply(context->m_memory_context, context->get_op_mem_ctxt(0), scalar);
    } break;
    case eOpcode::MAC: {
      const auto& scalar = attrs.getAttribute<D>(OP_ATTR_SCALAR);
      m_compute_backend->mac(
        context->m_memory_context, context->get_op_mem_ctxt(0), context->get_op_mem_ctxt(1), scalar);
    } break;
    case eOpcode::QUOTIENT:
      m_compute_backend->quotient(context->m_memory_context, context->get_op_mem_ctxt(0), context->get_op_mem_ctxt(1));
      break;
    case eOpcode::REMAINDER:
      m_compute_backend->remainder(context->m_memory_context, context->get_op_mem_ctxt(0), context->get_op_mem_ctxt(1));
      break;
    case eOpcode::DIV_BY_VANISHING: {
      const auto degree = attrs.getAttribute<uint64_t>(OP_ATTR_DEGREE);
      m_compute_backend->divide_by_vanishing_polynomial(context->m_memory_context, context->get_op_mem_ctxt(0), degree);
    } break;
    case eOpcode::ADD_MONOMIAL_INPLACE: {
      const auto monomial_coeff = attrs.getAttribute<C>("monomial_coeff");
      const auto monomial = attrs.getAttribute<uint64_t>("monomial");
      m_compute_backend->add_monomial_inplace(context->m_memory_context, monomial_coeff, monomial);
    } break;
    case eOpcode::SUB_MONOMIAL_INPLACE: {
      const auto monomial_coeff = attrs.getAttribute<C>("monomial_coeff");
      const auto monomial = attrs.getAttribute<uint64_t>("monomial");
      m_compute_backend->sub_monomial_inplace(context->m_memory_context, monomial_coeff, monomial);
    } break;
    case eOpcode::SLICE: {
      const auto offset = attrs.getAttribute<uint64_t>(OP_ATTR_OFFSET);
      const auto stride = attrs.getAttribute<uint64_t>(OP_ATTR_STRIDE);
      const auto size = attrs.getAttribute<uint64_t>(OP_ATTR_SIZE);
      m_compute_backend->slice(context->m_memory_context, context->get_op_mem_ctxt(0), offset, stride, size);
    } break;
    default:
      throw std::runtime_error("not implemented");
    }

    context->clear_operands(); // release ownership of operands
  }

  template <typename C, typename D, typename I>
  void Interpreter<C, D, I>::run(std::shared_ptr<TracingPolynomialContext<C, D, I>>& context)
  {
    this->m_visited.clear();
    evaluate(context);
  }

  template class Interpreter<>;

} // namespace polynomials