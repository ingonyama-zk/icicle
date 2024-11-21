#pragma once

#include <memory>
#include <vector>

namespace icicle {

/**
 * @brief Enum to represent all the operation supported by the Symbol class.
 *
 */
enum OpCode{
  OP_ADD = 0,
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
 * It saves oll the data to describe the operation done and point to its operands.
 * At any given time a Sybol contains an Operation instance that represent all the 
 * calculation history done so far to reach the current value.
 */

template <typename S>
class Operation {
  public:
    OpCode      m_opcode;
    std::shared_ptr<Operation<S> >  m_operand1; // 1st operand if exist
    std::shared_ptr<Operation<S> >  m_operand2; // 2nd operand if exist

    // optinal parameters:
    std::unique_ptr<S>          m_constant; // for OP_CONST: const value

    // implementation:
    int                         m_mem_addr; // location at the input vectors  

    // Constructor
    Operation<S> (OpCode opcode, 
                  std::shared_ptr<Operation<S> > operand1, 
                  std::shared_ptr<Operation<S> > operand2 = nullptr, 
                  std::unique_ptr<S> constant = nullptr,
                  int mem_addr = -1) :
      m_opcode(opcode),
      m_operand1(operand1),
      m_operand2(operand2),
      m_mem_addr(mem_addr),
      m_constant(std::move(constant)) {}


    bool was_visited(bool set_as_visit) {
      const bool was_visited = (m_visit_idx == s_last_visit);
      if (set_as_visit) {
        m_visit_idx = s_last_visit;
      }
      return was_visited;
    }

    static void reset_visit() {
      s_last_visit++;
    }

  private:
    unsigned int m_visit_idx = 0;
    static inline unsigned int s_last_visit = 1;
};

/**
 * @brief The basic variable used to describe a program.
 *
 * This class is used by the end user to describe a functionality later on provided
 * to execute by different backends.
 * This enables the user to constructs lamda.
 */
template <typename S>
class Symbol {
  public:
    // constructor
    Symbol() : m_operation(nullptr) {}

    // copy constructor
    Symbol(const Symbol& other) : m_operation(other.m_operation) {}

    // constructor init
    Symbol(const S& constant) : m_operation(std::make_shared<Operation<S> >(OpCode::OP_CONST, nullptr, nullptr, std::make_unique<S>(constant))) {}

    // operands
    Symbol operator=(const Symbol& operand) { return assign(operand);}
    Symbol operator=(const S& operand) { return assign(Symbol (operand));}
    Symbol operator+(const Symbol& operand) const { return add(operand);}
    Symbol operator+(const S& operand) const { return add(Symbol (operand));}
    Symbol operator+=(const Symbol& operand) { return assign(add(operand));}
    Symbol operator+=(const S& operand) { return assign(add(Symbol(operand)));}
    Symbol operator-(const Symbol& operand) const { return sub(operand);}
    Symbol operator-(const S& operand) const { return sub(Symbol(operand));}
    Symbol operator-=(const Symbol& operand) { return assign(sub(operand));}
    Symbol operator-=(const S& operand) { return assign(sub(Symbol(operand)));}
    Symbol operator*(const Symbol& operand) const { return multiply(operand);}
    Symbol operator*(const S& operand) const { return multiply(Symbol(operand));}
    Symbol operator*=(const Symbol& operand) { return assign(multiply(operand));}
    Symbol operator*=(const S& operand) { return assign(multiply(Symbol(operand)));}
    Symbol operator!() const { return inverse();}

    void set_as_input(int input_idx) {
      m_operation = std::make_shared<Operation<S> >(OP_INPUT, nullptr, nullptr, nullptr, input_idx);
    }

    // assign
    Symbol assign(const Symbol& other) {
      m_operation = other.m_operation;
      return *this;
    }

    // add
    Symbol add(const Symbol& operand) const {
      Symbol rv;
      rv.m_operation = std::make_shared<Operation<S> >(OP_ADD, m_operation, operand.m_operation);
      return rv;
    }

    // multiply
    Symbol multiply(const Symbol& operand) const {
      Symbol rv;
      rv.m_operation = std::make_shared<Operation<S> >(OP_MULT, m_operation, operand.m_operation);
      return rv;
    }

    // sub 
    Symbol sub(const Symbol& operand) const {
      Symbol rv;
      rv.m_operation = std::make_shared<Operation<S> >(OP_SUB, m_operation, operand.m_operation);
      return rv;
    }

    // inverse
    Symbol inverse() {
      Symbol rv;
      rv.m_operation = std::make_shared<Operation<S> >(OP_INV, m_operation);
      return rv;
    }

    std::shared_ptr<Operation< S > > m_operation;
};

} // namespace icicle