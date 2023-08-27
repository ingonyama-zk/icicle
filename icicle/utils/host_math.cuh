#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace host_math {

  // return x + y with uint32_t operands
  static __host__ uint32_t add(const uint32_t x, const uint32_t y) { return x + y; }

  // return x + y + carry with uint32_t operands
  static __host__ uint32_t addc(const uint32_t x, const uint32_t y, const uint32_t carry) { return x + y + carry; }

  // return x + y and carry out with uint32_t operands
  static __host__ uint32_t add_cc(const uint32_t x, const uint32_t y, uint32_t& carry)
  {
    uint32_t result;
    result = x + y;
    carry = x > result;
    return result;
  }

  // return x + y + carry and carry out  with uint32_t operands
  static __host__ uint32_t addc_cc(const uint32_t x, const uint32_t y, uint32_t& carry)
  {
    const uint32_t result = x + y + carry;
    carry = carry && x >= result || !carry && x > result;
    return result;
  }

  // return x - y with uint32_t operands
  static __host__ uint32_t sub(const uint32_t x, const uint32_t y) { return x - y; }

  // 	return x - y - borrow with uint32_t operands
  static __host__ uint32_t subc(const uint32_t x, const uint32_t y, const uint32_t borrow) { return x - y - borrow; }

  //	return x - y and borrow out with uint32_t operands
  static __host__ uint32_t sub_cc(const uint32_t x, const uint32_t y, uint32_t& borrow)
  {
    uint32_t result;
    result = x - y;
    borrow = x < result;
    return result;
  }

  //	return x - y - borrow and borrow out with uint32_t operands
  static __host__ uint32_t subc_cc(const uint32_t x, const uint32_t y, uint32_t& borrow)
  {
    const uint32_t result = x - y - borrow;
    borrow = borrow && x <= result || !borrow && x < result;
    return result;
  }

  // return x * y + z + carry and carry out with uint32_t operands
  static __host__ uint32_t madc_cc(const uint32_t x, const uint32_t y, const uint32_t z, uint32_t& carry)
  {
    uint32_t result;
    uint64_t r = static_cast<uint64_t>(x) * y + z + carry;
    carry = r >> 32;
    result = r & 0xffffffff;
    return result;
  }

  template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false>
  struct carry_chain {
    unsigned index;

    constexpr __host__ __forceinline__ carry_chain() : index(0) {}

    __host__ __forceinline__ uint32_t add(const uint32_t x, const uint32_t y, uint32_t& carry)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return host_math::add(x, y);
      else if (index == 1 && !CARRY_IN)
        return host_math::add_cc(x, y, carry);
      else if (index < OPS_COUNT || CARRY_OUT)
        return host_math::addc_cc(x, y, carry);
      else
        return host_math::addc(x, y, carry);
    }

    __host__ __forceinline__ uint32_t sub(const uint32_t x, const uint32_t y, uint32_t& carry)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return host_math::sub(x, y);
      else if (index == 1 && !CARRY_IN)
        return host_math::sub_cc(x, y, carry);
      else if (index < OPS_COUNT || CARRY_OUT)
        return host_math::subc_cc(x, y, carry);
      else
        return host_math::subc(x, y, carry);
    }
  };
} // namespace host_math
