#pragma once

#include <vector>
#include <functional>
#include "icicle/program/symbol.h"
#include "icicle/program/program.h"

namespace icicle {

  template <typename S>
  class CpuProgramExecutor
  {
  public:
    CpuProgramExecutor(const Program<S>& program)
        : m_program(program), m_variable_ptrs(program.get_nof_vars()), m_intermediates(program.m_nof_intermidiates)
    {
      // initialize m_variable_ptrs vector
      int variable_ptrs_idx = program.m_nof_inputs + program.m_nof_outputs;
      for (int idx = 0; idx < program.m_nof_constants; ++idx) {
        m_variable_ptrs[variable_ptrs_idx++] = (S*)(&(program.m_constants[idx]));
      }
      for (int idx = 0; idx < program.m_nof_intermidiates; ++idx) {
        m_variable_ptrs[variable_ptrs_idx++] = &(m_intermediates[idx]);
      }
    }

    // execute the program an return a program to the result
    void execute()
    {
      const std::byte* instruction;
      for (InstructionType instruction : m_program.m_instructions) {
        const int func_select = Program<S>::get_opcode(instruction);
        (this->*m_function_arr[func_select])(instruction);
      }
    }

    std::vector<S*> m_variable_ptrs;

  private:
    const Program<S>& m_program;
    std::vector<S> m_intermediates;

    // exe functions
    void exe_add(const InstructionType instruction)
    {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      *m_variable_ptrs[(int)inst_arr[Program<S>::INST_RESULT]] =
        *m_variable_ptrs[(int)inst_arr[Program<S>::INST_OPERAND1]] +
        *m_variable_ptrs[(int)inst_arr[Program<S>::INST_OPERAND2]];
    }

    void exe_mult(const InstructionType instruction)
    {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      *m_variable_ptrs[(int)inst_arr[Program<S>::INST_RESULT]] =
        *m_variable_ptrs[(int)inst_arr[Program<S>::INST_OPERAND1]] *
        *m_variable_ptrs[(int)inst_arr[Program<S>::INST_OPERAND2]];
    }

    void exe_sub(const InstructionType instruction)
    {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      *m_variable_ptrs[(int)inst_arr[Program<S>::INST_RESULT]] =
        *m_variable_ptrs[(int)inst_arr[Program<S>::INST_OPERAND1]] -
        *m_variable_ptrs[(int)inst_arr[Program<S>::INST_OPERAND2]];
    }

    void exe_inverse(const InstructionType instruction)
    {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      *m_variable_ptrs[(int)inst_arr[Program<S>::INST_RESULT]] =
        S::inverse(*m_variable_ptrs[(int)inst_arr[Program<S>::INST_OPERAND1]]);
    }

    void exe_predef_ab_minus_c(const InstructionType instruction)
    {
      const S& a = *m_variable_ptrs[0];
      const S& b = *m_variable_ptrs[1];
      const S& c = *m_variable_ptrs[2];
      *m_variable_ptrs[3] = a * b - c;
    }
    void exe_predef_eq_x_ab_minus_c(const InstructionType instruction)
    {
      const S& a = *m_variable_ptrs[0];
      const S& b = *m_variable_ptrs[1];
      const S& c = *m_variable_ptrs[2];
      const S& eq = *m_variable_ptrs[3];
      *m_variable_ptrs[4] = eq * (a * b - c);
    }

    using FunctionPtr = void (CpuProgramExecutor::*)(const InstructionType);
    inline static const FunctionPtr m_function_arr[] = {
      &CpuProgramExecutor::exe_add,     // OP_ADD
      &CpuProgramExecutor::exe_mult,    // OP_MULT
      &CpuProgramExecutor::exe_sub,     // OP_SUB
      &CpuProgramExecutor::exe_inverse, // OP_INV
      // pre defined functions
      &CpuProgramExecutor::exe_predef_ab_minus_c,       // predef A*B-C
      &CpuProgramExecutor::exe_predef_eq_x_ab_minus_c}; // predef EQ*(A*B-C)
  };

} // namespace icicle