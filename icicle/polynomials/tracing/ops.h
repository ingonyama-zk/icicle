#pragma once

#include <string>
#include <unordered_map>
#include <variant>

namespace polynomials {

  // Enum for identifying different polynomial operations.
  enum eOpcode {
    FROM_COEFFS = 0,
    FROM_ROU_EVALS,
    CLONE,
    ADD,
    SUB,
    MUL,
    DIV,
    QUOTIENT,
    REMAINDER,
    DIV_BY_VANISHING,
    ADD_MONOMIAL_INPLACE,
    SUB_MONOMIAL_INPLACE,
    SLICE,
  };

  // Class for storing and managing attributes associated with polynomial operations.
  // Supported types are int and std::string
  class Attributes
  {
  public:
    template <typename T>
    void setAttribute(const std::string& name, const T& value)
    {
      attributes[name] = value;
    }

    template <typename T>
    T getAttribute(const std::string& name) const
    {
      if (attributes.find(name) == attributes.end()) { throw std::runtime_error("Attribute not found"); }

      return std::get<T>(attributes.at(name));
    }

  private:
    std::unordered_map<std::string, std::variant<int, std::string>> attributes;
  };

  // Struct representing a polynomial operation with an opcode and associated attributes.
  struct Op {
    eOpcode opcode;        // Opcode indicating the type of operation.
    Attributes attributes; // Attributes providing additional information or parameters for the operation.
  };

} // namespace polynomials
