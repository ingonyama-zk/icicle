#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <sstream>
#include "fields/field_config.cuh"

using field_config::scalar_t;
namespace polynomials {

  // Enum for identifying different polynomial operations.
  enum eOpcode {
    FROM_COEFFS = 0,
    FROM_ROU_EVALS,
    CLONE,
    ADD,
    SUB,
    MUL,
    SCALAR_MUL,
    DIV,
    QUOTIENT,
    REMAINDER,
    DIV_BY_VANISHING,
    ADD_MONOMIAL_INPLACE,
    SUB_MONOMIAL_INPLACE,
    SLICE,

    INVALID,
    NOF_OPCODES,
  };

  static const char* OpcodeToStr(eOpcode opcode)
  {
    static const char* s_op_str[eOpcode::NOF_OPCODES] = {
      "FROM_COEFFS",
      "FROM_ROU_EVALS",
      "CLONE",
      "ADD",
      "SUB",
      "MUL",
      "SCALAR_MUL",
      "DIV",
      "QUOTIENT",
      "REMAINDER",
      "DIV_BY_VANISHING",
      "ADD_MONOMIAL_INPLACE",
      "SUB_MONOMIAL_INPLACE",
      "SLICE",
      "INVALID",
    };

    return s_op_str[opcode];
  }

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

    std::string to_string() const
    {
      std::ostringstream oss;
      for (const auto& [key, val] : attributes) {
        oss << key << "=" << visit_variant(val) << "\n";
      }
      return oss.str();
    }

  private:
    std::unordered_map<std::string, std::variant<int64_t, uint64_t, std::string, scalar_t>> attributes;

    static std::string visit_variant(const std::variant<int64_t, uint64_t, std::string, scalar_t>& var)
    {
      return std::visit(
        [](auto&& arg) -> std::string {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::string>) {
            return arg; // No conversion needed for strings.
          } else if constexpr (std::is_same_v<T, scalar_t>) {
            return "...";
          } else {
            return std::to_string(arg); // Convert numeric types to string.
          }
        },
        var);
    }
  };

#define OP_ATTR_DEGREE "degree"
#define OP_ATTR_OFFSET "offset"
#define OP_ATTR_STRIDE "stride"
#define OP_ATTR_SIZE   "size"
#define OP_ATTR_SCALAR "scalar"

} // namespace polynomials
