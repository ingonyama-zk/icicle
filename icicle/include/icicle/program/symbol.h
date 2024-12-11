#pragma once

#include <memory>
#include <vector>

namespace icicle {

  /**
   * @brief Enum to represent all the operation supported by the Symbol class.
   *
   */
  enum ProgramOpcode {
    OP_COPY = 0,
    OP_ADD,
    OP_MULT,
    OP_SUB,
    OP_INV,

    NOF_OPERATIONS,

    OP_INPUT,
    OP_CONST,
  };

  /**
   * @brief a node at the data-flow grapsh (DFG) that represent a single operation
   *
   * This class contains a data about an operation done by the Symbol class.
   * It saves all the data to describe the operation done and points to its operands.
   * At any given time a Symbol contains an Operation instance that represent all the
   * calculation history done so far to reach the current value.
   */

  template <typename S>
  class Operation
  {
  public:
    ProgramOpcode m_opcode;
    std::shared_ptr<Operation<S>> m_operand1; // 1st operand if exist
    std::shared_ptr<Operation<S>> m_operand2; // 2nd operand if exist

    // optional parameters:
    std::unique_ptr<S> m_constant; // for OP_CONST: const value
    int m_poly_degree;             // number of multiplications so far

    // implementation:
    int m_variable_idx; // location at the intermediate variables vectors

    // Constructor
    Operation<S>(
      ProgramOpcode opcode,
      std::shared_ptr<Operation<S>> operand1,
      std::shared_ptr<Operation<S>> operand2 = nullptr,
      std::unique_ptr<S> constant = nullptr,
      int variable_idx = -1)
        : m_opcode(opcode), m_operand1(operand1), m_operand2(operand2), m_variable_idx(variable_idx),
          m_constant(std::move(constant))
    {
      update_poly_degree();
    }

    bool is_visited(bool set_as_visit)
    {
      const bool is_visited = (m_visit_idx == s_last_visit); // s_last_visit was not incremented since visited
      if (set_as_visit) {
        m_visit_idx = s_last_visit; // set operation as visited
      }
      return is_visited;
    }

    // reset visit for all operations
    static void reset_visit()
    {
      // changing s_last_visit means that for all operation m_visit_idx != s_last_visit
      s_last_visit++;
    }

  private:
    unsigned int m_visit_idx = 0;
    static inline unsigned int s_last_visit = 1;

    // update the current poly_degree based on the operands
    void update_poly_degree()
    {
      // if one of the operand has undef poly_degree
      if ((m_operand1 && m_operand1->m_poly_degree < 0) || (m_operand2 && m_operand2->m_poly_degree < 0)) {
        m_poly_degree = -1;
        return;
      }
      switch (m_opcode) {
      case OP_ADD:
      case OP_SUB:
        m_poly_degree = std::max(m_operand1->m_poly_degree, m_operand2->m_poly_degree);
        return;
      case OP_MULT:
        m_poly_degree = m_operand1->m_poly_degree + m_operand2->m_poly_degree;
        return;
      case OP_INV:
        m_poly_degree = -1; // undefined
        return;
      default:
        m_poly_degree = 0;
      }
    }
  };

  /**
   * @brief The basic variable used to describe a program.
   *
   * This class is used by the end user to describe a functionality later on provided
   * to execute by different backends.
   * This enables the user to constructs lambda.
   */
  template <typename S>
  class Symbol
  {
  public:
    // constructor
    Symbol() : m_operation(nullptr) {}

    // copy constructor
    Symbol(const Symbol& other) : m_operation(other.m_operation) {}

    // constructor init
    Symbol(const S& constant)
        : m_operation(
            std::make_shared<Operation<S>>(ProgramOpcode::OP_CONST, nullptr, nullptr, std::make_unique<S>(constant)))
    {
    }

    // operands
    Symbol operator=(const Symbol& operand) { return assign(operand); }
    Symbol operator=(const S& operand) { return assign(Symbol(operand)); }
    Symbol operator+(const Symbol& operand) const { return add(operand); }
    Symbol operator+(const S& operand) const { return add(Symbol(operand)); }
    Symbol operator+=(const Symbol& operand) { return assign(add(operand)); }
    Symbol operator+=(const S& operand) { return assign(add(Symbol(operand))); }
    Symbol operator-(const Symbol& operand) const { return sub(operand); }
    Symbol operator-(const S& operand) const { return sub(Symbol(operand)); }
    Symbol operator-=(const Symbol& operand) { return assign(sub(operand)); }
    Symbol operator-=(const S& operand) { return assign(sub(Symbol(operand))); }
    Symbol operator*(const Symbol& operand) const { return multiply(operand); }
    Symbol operator*(const S& operand) const { return multiply(Symbol(operand)); }
    Symbol operator*=(const Symbol& operand) { return assign(multiply(operand)); }
    Symbol operator*=(const S& operand) { return assign(multiply(Symbol(operand))); }

    // inverse
    Symbol inverse() const
    {
      Symbol rv;
      rv.m_operation = std::make_shared<Operation<S>>(OP_INV, m_operation);
      return rv;
    }

    void set_as_input(int input_idx)
    {
      m_operation = std::make_shared<Operation<S>>(OP_INPUT, nullptr, nullptr, nullptr, input_idx);
    }

    // assign
    Symbol assign(const Symbol& other)
    {
      m_operation = other.m_operation;
      return *this;
    }

    // add
    Symbol add(const Symbol& operand) const
    {
      Symbol rv;
      rv.m_operation = std::make_shared<Operation<S>>(OP_ADD, m_operation, operand.m_operation);
      return rv;
    }

    // multiply
    Symbol multiply(const Symbol& operand) const
    {
      Symbol rv;
      rv.m_operation = std::make_shared<Operation<S>>(OP_MULT, m_operation, operand.m_operation);
      return rv;
    }

    // sub
    Symbol sub(const Symbol& operand) const
    {
      Symbol rv;
      rv.m_operation = std::make_shared<Operation<S>>(OP_SUB, m_operation, operand.m_operation);
      return rv;
    }

    std::shared_ptr<Operation<S>> m_operation;
  };

} // namespace icicle