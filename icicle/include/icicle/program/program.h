#pragma once

#include <vector>
#include <functional>
#include "icicle/program/symbol.h"

namespace icicle {

  using InstructionType = uint32_t;

  enum PreDefinedPrograms {
    AB_MINUS_C = 0, // (A*B)-C
    EQ_X_AB_MINUS_C // E*(A*B-C)
  };

  /**
   * @brief A class that convert the function described by user into a program that can be executed.
   *
   * This class receives a Symbol instance that contains a DFG representing the required calculation.
   * It generates a vector of instructions that represent the calculation.
   * Each instruction has the following format.
   * bits 7:0   - opcode according to enum ProgramOpcode
   * bits 15:8  - operand 1 selector from the input vector
   * bits 23:16 - operand 2 selector from the input vector
   * bits 31:24 - result selector
   */
  template <typename S>
  class Program
  {
  public:
    // Generate a program based on a lambda function with multiple inputs and multiple outputs
    Program(std::function<void(std::vector<Symbol<S>>&)> program_func, const int nof_parameters)
        : m_nof_parameters(nof_parameters)
    {
      // Generate the DFG
      std::vector<Symbol<S>> program_parameters(m_nof_parameters);
      set_as_inputs(program_parameters);
      program_func(program_parameters);

      // based on the DFG, generate the instructions
      generate_program(program_parameters);
    }

    // Generate a program based on a PreDefinedPrograms
    Program(PreDefinedPrograms pre_def)
    {
      // set number of parameters
      switch (pre_def) {
      case AB_MINUS_C:
        m_nof_parameters = 4;
        break;
      case EQ_X_AB_MINUS_C:
        m_nof_parameters = 5;
        break;
      default:
        ICICLE_LOG_ERROR << "Illegal opcode: " << int(pre_def);
      }
      // build the instruction
      int instruction = int(ProgramOpcode::NOF_OPERATIONS) + int(pre_def);
      m_instructions.push_back(instruction);
    }

    // run over all inputs at the vector and set their operands to OP_INPUT
    void set_as_inputs(std::vector<Symbol<S>>& program_parameters)
    {
      for (int parameter_idx = 0; parameter_idx < program_parameters.size(); parameter_idx++) {
        program_parameters[parameter_idx].set_as_input(parameter_idx);
      }
    }

    // run over the DFG held by program_parameters and generate the program
    void generate_program(std::vector<Symbol<S>>& program_parameters)
    {
      // run over the graph and allocate location for all constants
      Operation<S>::reset_visit();
      for (auto& result : program_parameters) {
        allocate_constants(result.m_operation);
      }

      // run over the graph and generate the program
      Operation<S>::reset_visit();
      for (int parameter_idx = 0; parameter_idx < program_parameters.size(); parameter_idx++) {
        Symbol<S>& cur_symbol = program_parameters[parameter_idx];
        // the operation not yet calculated
        if (cur_symbol.m_operation->m_variable_idx == -1) {
          // set the operation location to the current parameter_idx
          cur_symbol.m_operation->m_variable_idx = parameter_idx;
          generate_program(cur_symbol.m_operation);
          continue;
        }
        // the operation already calculated but on a different location
        if (cur_symbol.m_operation->m_variable_idx != parameter_idx)
          // copy it to this parameter
          push_copy_instruction(cur_symbol.m_operation->m_variable_idx, parameter_idx);
      }
    }

    // Program
    std::vector<InstructionType> m_instructions; // vector of instructions to execute
    std::vector<S> m_constants;                  // vector of constants to use
    int m_nof_parameters = 0;
    int m_nof_constants = 0;
    int m_nof_intermidiates = 0;

    const int get_nof_vars() const { return m_nof_parameters + m_nof_constants + m_nof_intermidiates; }

    static inline const int INST_OPCODE = 0;
    static inline const int INST_OPERAND1 = 1;
    static inline const int INST_OPERAND2 = 2;
    static inline const int INST_RESULT = 3;
    inline static int get_opcode(const InstructionType instruction) { return (instruction & 0xFF); }

  protected:
    // default constructor
    Program() {}

    // run recursively on the DFG and push instruction per operation
    void generate_program(std::shared_ptr<Operation<S>> operation)
    {
      if (
        operation == nullptr || operation->is_visited(true) || operation->m_opcode == OP_INPUT ||
        operation->m_opcode == OP_CONST)
        return;
      generate_program(operation->m_operand1);
      generate_program(operation->m_operand2);

      push_instruction(operation);
    }

    // run over the DFG and collectall the constants.
    void allocate_constants(std::shared_ptr<Operation<S>> operation)
    {
      if (operation == nullptr || operation->is_visited(true)) return;
      allocate_constants(operation->m_operand1);
      allocate_constants(operation->m_operand2);
      // if constant located
      if (operation->m_opcode == OP_CONST) {
        m_constants.push_back(*(operation->m_constant)); // push it to constant vector
        operation->m_variable_idx = allocate_constant(); // set its location after the parameters
      }
    }

    int allocate_constant() { return (m_nof_parameters + m_nof_constants++); }
    int allocate_intermidiate() { return (m_nof_parameters + m_nof_constants + m_nof_intermidiates++); }

    // Build an instruction
    void push_instruction(std::shared_ptr<Operation<S>> operation)
    {
      // Build an instruction on the array
      std::byte int_arr[sizeof(InstructionType)] = {};
      // Set instruction::opcode
      int_arr[INST_OPCODE] = std::byte(operation->m_opcode);
      // Set instruction::operand1
      int_arr[INST_OPERAND1] = std::byte(operation->m_operand1->m_variable_idx);

      if (operation->m_operand2) {
        // Set instruction::operand2
        int_arr[INST_OPERAND2] = std::byte(operation->m_operand2->m_variable_idx);
      }

      if (operation->m_variable_idx < 0) {
        // allocate a register for the result
        operation->m_variable_idx = allocate_intermidiate();
      }

      // Set instruction::operand2
      int_arr[INST_RESULT] = std::byte(operation->m_variable_idx);
      InstructionType instruction;
      std::memcpy(&instruction, int_arr, sizeof(InstructionType));
      m_instructions.push_back(instruction);
    }

    void push_copy_instruction(const int source_idx, const int dest_idx)
    {
      // Build an instruction on the array
      std::byte int_arr[sizeof(InstructionType)] = {};
      // Set instruction::opcode
      int_arr[INST_OPCODE] = std::byte(ProgramOpcode::OP_COPY);
      // Set instruction::operand1
      int_arr[INST_OPERAND1] = std::byte(source_idx);

      // Set instruction::operand2
      int_arr[INST_RESULT] = std::byte(dest_idx);
      InstructionType instruction;
      std::memcpy(&instruction, int_arr, sizeof(InstructionType));
      m_instructions.push_back(instruction);
    }

  public:
    void print_program()
    {
      std::cout << "nof_parameters: " << m_nof_parameters << std::endl;
      std::cout << "nof_constants: " << m_nof_constants << std::endl;
      std::cout << "nof_intermidiates: " << m_nof_intermidiates << std::endl;
      std::cout << "Constants:: " << std::endl;
      for (auto constant : m_constants) {
        std::cout << "   " << constant << std::endl;
      }
      std::cout << "Instructions:: " << std::endl;
      for (auto inst : m_instructions) {
        std::cout << "   Opcode: " << (inst & 0xFF) << ", op1: " << ((inst >> 8) & 0xFF)
                  << ", op2: " << ((inst >> 16) & 0xFF) << ", Res: " << ((inst >> 24) & 0xFF) << std::endl;
      }
    }
  };

} // namespace icicle
