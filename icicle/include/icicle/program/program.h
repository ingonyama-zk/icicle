#pragma once

#include <vector>
#include <functional>
#include "icicle/program/symbol.h"

namespace icicle {

using InstructionType = uint32_t;



enum PreDefinedPrograms{
  IDENTITY,
  AB_MINUS_C,
  EQ_X_AB_MINUS_C
};


/**
 * @brief A class that convert the function described by user into a program that can be executed
 *
 * This class recieves a Symbol instance that contains a DFG representing the required calculation.
 * It generates a vector of instructions that represent the calculation. 
 * Each instruction has the following format.
 * bits 7:0   - opcode according to enum OpCode
 * bits 15:8  - operand 1 selector from the input vector
 * bits 23:16 - operand 2 selector from the input vector
 * bits 31:24 - result selector
 */
template <typename S>
class Program {
public:
  // Generate a program based on a lambda function
  Program(std::function<Symbol<S> (std::vector< Symbol<S> >&)> program_func, const int nof_inputs) {
    std::vector<Symbol> program_inputs(nof_inputs);
    set_as_inputs(program_inputs);
    Symbol result = program_func(program_inputs);
    generate_program(result);
  }
  
  // Generate a program based on a PreDefinedPrograms
  Program(PreDefinedPrograms pre_def) {
    int instruction = int(OpCode::NOF_OPERATIONS) + int(pre_def);
    m_instructions.push_back(instruction);
  }

  // run over all symbols at the vector and set there gen process to OP_INPUT
  void set_as_inputs(std::vector<Symbol<S>>& combine_inputs) const {
    m_nof_inputs = combine_inputs.size();
    for (int idx = 0; idx < m_nof_inputs; idx++) {
      combine_inputs[input_idx].set_as_input(idx);
    }
  }

  // run over the DFG held by result and gemerate the program
  void generate_program(Symbol& result) {
    m_nof_outputs = 1;
    result.m_operation->m_mem_addr = m_nof_inputs;
    allocate_constants(result.m_operation);
    generate_program(result.m_operation);
  }

  // Program
  std::vector<InstructionType>   m_instructions;
  std::vector<S>                 m_constants;
  int                            m_nof_inputs = 0;
  int                            m_nof_outputs = 0;
  int                            m_nof_constants = 0;
  int                            m_nof_intermidiates = 0;

  const int get_nof_vars() const {
    return m_nof_inputs + m_nof_outputs + m_nof_constants + m_nof_intermidiates;
  }

private:
  void generate_program(std::shared_ptr<Operation< S > > operation) {
    if (operation == nullptr)
      return;
    generate_program(operation->m_operand1);
    generate_program(operation->m_operand2);

    InstructionType instruction(0);
    int result_addr = 
    switch(operation->m_opcode) {
      case OP_INV:
        instruction = int(operation->m_opcode);
        instruction |= operation->m_operand1->m_mem_addr << 8;
      case OP_ADD:
      case OP_MULT:
      case OP_SUB:
        instruction |= operation->m_operand2->m_mem_addr << 16;
        if (operation->m_mem_addr != 0) {
          operation->m_mem_addr = m_nof_inputs + m_nof_constants + m_nof_intermidiates++;
        }
        instruction |= operation->m_mem_addr << 24;
        m_instructions.push_back(instruction);
        return;
      case OP_INPUT:
      case OP_CONST:
        return;
      default:
        ICICLE_ERROR << "Unsupported operation opcode: " << operation->m_opcode << std::endl;
        ICICLE_ASSERT(false); // unsupported opcode
    }
  }

  void allocate_constants(std::shared_ptr<Operation<S> > operation) {
    if (operation == nullptr)
      return;
    allocate_constants(operation->m_operand1);
    allocate_constants(operation->m_operand2);
    if (operation->m_opcode == OP_CONST) {
      m_constants.push_back(*(operation->m_constant));
      operation->m_mem_addr = m_nof_inputs + m_nof_outputs + m_nof_constants;
      m_nof_constants++;
    }
  }

  void print_program() {
    for (auto inst : m_instructions) {
      std::cout << "Opcode: " << inst & 0xFF << ", op1: " << (inst >> 8) & 0xFF << 
        ", op2: " << (inst >> 16) & 0xFF << ", Res: " << (inst >> 24) & 0xFF << std::endl;
    }
  }
};


