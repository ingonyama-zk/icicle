#pragma once

#include <vector>
#include <functional>
#include "icicle/program/symbol.h"
#include "icicle/program/program.h"

namespace icicle {

template <typename S>
class CpuProgramExecutor {
  public:
    CpuProgramExecutor(Program<S>& program) : 
      m_program(program),
      m_variable_ptrs(program.get_nof_vars()),
      m_intermidites(program.m_nof_intermidiates) {
        // initialize m_variable_ptrs vector
        int variable_ptrs_idx = program.m_nof_inputs + program.m_nof_outputs;
        for (int idx=0; idx<program.m_nof_constants; ++idx) {
          m_variable_ptrs[variable_ptrs_idx++] = &(program.m_constants[idx]);
        }
        for (int idx=0; idx<program.m_nof_intermidiates; ++idx) {
          m_variable_ptrs[variable_ptrs_idx++] = &(m_intermidites[idx]);
        }
      }

    // execute the program an return a program to the result
    void execute () {
      const std::byte* instruction;
      for (InstructionType instruction : m_program.m_instructions) {
        const int func_select = (instruction & 0xFF);
        (this->*m_function_arr[instruction & 0xFF])(instruction);
      }
    }


  std::vector <S*> m_variable_ptrs;
  
  private:
    Program<S> m_program;
    std::vector <S>  m_intermidites;

    // exe functions
    void exe_add(const InstructionType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      *m_variable_ptrs[(int)inst_arr[3]] = *m_variable_ptrs[(int)inst_arr[1]] + *m_variable_ptrs[(int)inst_arr[2]]; 
    }


    void exe_mult(const InstructionType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      *m_variable_ptrs[(int)inst_arr[3]] = *m_variable_ptrs[(int)inst_arr[1]] * *m_variable_ptrs[(int)inst_arr[2]]; 
    }

    void exe_sub(const InstructionType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      *m_variable_ptrs[(int)inst_arr[3]] = *m_variable_ptrs[(int)inst_arr[1]] - *m_variable_ptrs[(int)inst_arr[2]]; 
    }

    void exe_inverse(const InstructionType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      *m_variable_ptrs[(int)inst_arr[3]] = S::inverse(*m_variable_ptrs[(int)inst_arr[1]]); 
    }

    void exe_predef_ab_minus_c(const InstructionType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      const S& a = *m_variable_ptrs[0];
      const S& b = *m_variable_ptrs[1];
      const S& c = *m_variable_ptrs[2];
      *m_variable_ptrs[3] = a*b - c;
    }
    void exe_predef_eq_x_ab_minus_c(const InstructionType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      const S& a = *m_variable_ptrs[0];
      const S& b = *m_variable_ptrs[1];
      const S& c = *m_variable_ptrs[2];
      const S& eq = *m_variable_ptrs[3];
      *m_variable_ptrs[4] = eq*(a*b - c);
    }

    using FunctionPtr = void(CpuProgramExecutor::*)(const InstructionType);
    inline static const FunctionPtr m_function_arr[] = {
      &CpuProgramExecutor::exe_add,  // OP_ADD
      &CpuProgramExecutor::exe_mult, // OP_MULT
      &CpuProgramExecutor::exe_sub,  // OP_SUB
      &CpuProgramExecutor::exe_inverse,   // OP_INV      
      // pre defined functions
      &CpuProgramExecutor::exe_predef_ab_minus_c,
      &CpuProgramExecutor::exe_predef_eq_x_ab_minus_c
    };
};

} // namespace icicle