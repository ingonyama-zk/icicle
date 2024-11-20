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
      m_variable_ptrs(program.get_nof_vars())
      m_intermidites(program.m_nof_intermidiates) {
        // initialize m_variable_ptrs vector
        int variable_ptrs_idx = program.m_nof_inputs;
        for (int idx=0; idx<program.m_nof_constants; ++idx) {
          m_variable_ptrs[variable_ptrs_idx] = &(program.m_constants[idx]);
          variable_ptrs_idx++;
        }
        for (int idx=0; idx<program.m_nof_intermidiates; ++idx) {
          m_variable_ptrs[variable_ptrs_idx] = &(program.m_constants[idx]);
          variable_ptrs_idx++;
        }
      }

    // execute the program an return a program to the result
    void execute () {
      const std::byte* instruction;
      for (InstType instruction : m_program.m_instructions) {
        inst_arr = reinterpret_cast<const std::byte*>(&instruction);
        (*functionPtrs[inst_arr[0]])();
      }
    }


  std::vector <S*> m_variable_ptrs;
  
  private:
    Program<S> m_instructions;
    std::vector <S>  m_intermidites;

    // exe functions
    void exe_add(const InstType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      m_variable_ptrs[inst_arr[3]] = m_variable_ptrs[inst_arr[2]] + m_variable_ptrs[inst_arr[1]]; 
    }


    void exe_mult(const InstType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      m_variable_ptrs[inst_arr[3]] = m_variable_ptrs[inst_arr[2]] * m_variable_ptrs[inst_arr[1]]; 
    }

    void exe_sub(const InstType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      m_variable_ptrs[inst_arr[3]] = m_variable_ptrs[inst_arr[2]] - m_variable_ptrs[inst_arr[1]]; 
    }

    void exe_inv(const InstType instruction) {
      const std::byte* inst_arr = reinterpret_cast<const std::byte*>(&instruction);
      m_variable_ptrs[inst_arr[3]] = inv(m_variable_ptrs[inst_arr[1]]); 
    }

    void exe_predef_identity(const InstType instruction) {
    }
    void exe_predef_ab_minus_c(const InstType instruction) {
      const S& a = m_variable_ptrs[inst_arr[0]];
      const S& b = m_variable_ptrs[inst_arr[1]];
      const S& c = m_variable_ptrs[inst_arr[2]];
      m_variable_ptrs[inst_arr[3]] = a*b - c;
    }
    void exe_predef_eq_x_ab_minus_c(const InstType instruction) {
      const S& a = m_variable_ptrs[inst_arr[0]];
      const S& b = m_variable_ptrs[inst_arr[1]];
      const S& c = m_variable_ptrs[inst_arr[2]];
      const S& eq = m_variable_ptrs[inst_arr[3]];
      m_variable_ptrs[inst_arr[4]] = eq*(a*b - c);
    }

    using FunctionPtr = void(*)(const InstType);
    static constexpr std::array<FunctionPtr, static_cast<int>(NOF_OPERATIONS)> functionPtrs = {
      &exe_add,  // OP_ADD
      &exe_mult, // OP_MULT
      &exe_sub,  // OP_SUB
      &exe_inv,   // OP_INV      
      // pre defined functions
      &exe_predef_identity,
      &exe_predef_ab_minus_c,
      &exe_predef_eq_x_ab_minus_c
    };
};

} // namespace icicle