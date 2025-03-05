#pragma once

#include <vector>
#include <functional>
#include "icicle/program/symbol.h"
#include "icicle/program/program.h"

namespace icicle {

  /**
   * @brief A class that convert the function with inputs and return value described by user into a program that can be
   * executed.
   */

  template <typename S>
  class ReturningValueProgram : public Program<S>
  {
  public:
    // Generate a program based on a lambda function with multiple inputs and 1 output as a return value
    ReturningValueProgram(std::function<Symbol<S>(std::vector<Symbol<S>>&)> program_func, int nof_inputs)
    {
      this->m_nof_parameters = nof_inputs + 1;
      std::vector<Symbol<S>> program_parameters(this->m_nof_parameters);
      this->set_as_inputs(program_parameters);
      program_parameters[nof_inputs] = program_func(program_parameters); // place the output after the all inputs
      this->generate_program(program_parameters);
    }

    // Generate a program based on a PreDefinedPrograms
    ReturningValueProgram(PreDefinedPrograms pre_def) : Program<S>(pre_def)
    {
      switch (pre_def) {
      case AB_MINUS_C:
        m_poly_degree = 2;
        break;
      case EQ_X_AB_MINUS_C:
        m_poly_degree = 3;
        break;
      default:
        ICICLE_LOG_ERROR << "Illegal opcode: " << int(pre_def);
      }
    }

    // Call base generate_program as well as updating the required polynomial degree
    void generate_program(std::vector<Symbol<S>>& program_parameters) override
    {
      Program<S>::generate_program(program_parameters);
      m_poly_degree = program_parameters.back().m_operation->m_poly_degree;
    }

    int get_polynomial_degree() const { return m_poly_degree; }

  protected:
    ReturningValueProgram() {}

    // Friend function for C-api to have access to the default constructor
    template <typename T>
    friend ReturningValueProgram<T>* create_empty_returning_value_program();

  private:
    int m_poly_degree = 0;
  };

  // Friend function for C-api to have access to the default constructor
  template <typename S>
  ReturningValueProgram<S>* create_empty_returning_value_program()
  {
    return new ReturningValueProgram<S>();
  }
} // namespace icicle
