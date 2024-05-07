#pragma once
#include <cstdint>

namespace ptx {

   uint32_t add(const uint32_t x, const uint32_t y)
  {
    uint32_t result = 0;
    return result;
  }

   uint32_t add_cc(const uint32_t x, const uint32_t y)
  {
    uint32_t result = 0;
    return result;
  }

   uint32_t addc(const uint32_t x, const uint32_t y)
  {
    uint32_t result = 0;
    return result;
  }

   uint32_t addc_cc(const uint32_t x, const uint32_t y)
  {
    uint32_t result = 0;
    return result;
  }

   uint32_t sub(const uint32_t x, const uint32_t y)
  {
    uint32_t result = 0;
    return result;
  }

   uint32_t sub_cc(const uint32_t x, const uint32_t y)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t subc(const uint32_t x, const uint32_t y)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t subc_cc(const uint32_t x, const uint32_t y)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t mul_lo(const uint32_t x, const uint32_t y)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t mul_hi(const uint32_t x, const uint32_t y)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t mad_lo(const uint32_t x, const uint32_t y, const uint32_t z)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t mad_hi(const uint32_t x, const uint32_t y, const uint32_t z)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t mad_lo_cc(const uint32_t x, const uint32_t y, const uint32_t z)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t mad_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t madc_lo(const uint32_t x, const uint32_t y, const uint32_t z)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t madc_hi(const uint32_t x, const uint32_t y, const uint32_t z)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t madc_lo_cc(const uint32_t x, const uint32_t y, const uint32_t z)
  {
     uint32_t result = 0;
    return result;
  }

   uint32_t madc_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z)
  {
     uint32_t result = 0;
    return result;
  }

   uint64_t mov_b64(uint32_t lo, uint32_t hi)
  {
    uint64_t result = 0;
    return result;
  }

  // Gives u64 overloads a dedicated namespace.
  // Callers should know exactly what they're calling (no implicit conversions).
  namespace u64 {

     uint64_t add(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t add_cc(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t addc(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t addc_cc(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t sub(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t sub_cc(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t subc(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t subc_cc(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t mul_lo(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t mul_hi(const uint64_t x, const uint64_t y)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t mad_lo(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t mad_hi(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t mad_lo_cc(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t mad_hi_cc(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t madc_lo(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t madc_hi(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t madc_lo_cc(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result = 0;
      return result;
    }

     uint64_t madc_hi_cc(const uint64_t x, const uint64_t y, const uint64_t z)
    {
      uint64_t result = 0;
      return result;
    }

  } // namespace u64

   void bar_arrive(const unsigned name, const unsigned count)
  {
    return;
  }

   void bar_sync(const unsigned name, const unsigned count)
  {
    return;
  }

} // namespace ptx