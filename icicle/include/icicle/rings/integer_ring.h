

#include "icicle/math/storage.h"
#include "icicle/math/host_math.h"

#ifdef __CUDA_ARCH__
namespace base_math = device_math;
#else
namespace base_math = host_math;
#endif

// TODO Yuval: cuda math

template <class CONFIG>
class IntegerRing
{
private:
  static constexpr unsigned TLC = CONFIG::limbs_count;
  static constexpr unsigned NBITS = CONFIG::modulus_bit_count;

  // TODO Yuval: ff is wrong but easier to bring code from field.h for now. Review this later
  using ff_storage = storage<TLC>;
  using ff_wide_storage = storage<2 * TLC>;

  ff_storage limbs_storage;

public:
  IntegerRing() {}
  explicit IntegerRing(const ff_storage& limbs) : limbs_storage(limbs) {}
  static constexpr HOST_DEVICE_INLINE IntegerRing zero() { return IntegerRing{CONFIG::zero}; }
  static constexpr HOST_DEVICE_INLINE IntegerRing one() { return IntegerRing{CONFIG::one}; }
  static constexpr HOST_DEVICE_INLINE IntegerRing from(uint32_t value)
  {
    ff_storage element{};
    element.limbs[0] = value;
    for (int i = 1; i < TLC; i++) {
      element.limbs[i] = 0;
    }
    return IntegerRing{element};
  }

  /* OPERATORS */

  // return modulus multiplied by 1, 2 or 4
  template <unsigned MULTIPLIER = 1>
  static constexpr HOST_DEVICE_INLINE ff_storage get_modulus()
  {
    switch (MULTIPLIER) {
    case 1:
      return CONFIG::modulus;
    case 2:
      return CONFIG::modulus_2;
    case 4:
      return CONFIG::modulus_4;
    default:
      return {};
    }
  }

  template <unsigned REDUCTION_SIZE = 1>
  static constexpr HOST_DEVICE_INLINE IntegerRing sub_modulus(const IntegerRing& xs)
  {
    if (REDUCTION_SIZE == 0) return xs;
    const ff_storage modulus = get_modulus<REDUCTION_SIZE>();
    IntegerRing rs = {};
    return sub_limbs<TLC, true>(xs.limbs_storage, modulus, rs.limbs_storage) ? xs : rs;
  }

  template <unsigned NLIMBS, bool CARRY_OUT>
  static constexpr HOST_DEVICE_INLINE uint32_t
  add_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    return base_math::template add_sub_limbs<NLIMBS, false, CARRY_OUT>(xs, ys, rs);
  }

  template <unsigned NLIMBS, bool CARRY_OUT>
  static constexpr HOST_DEVICE_INLINE uint32_t
  sub_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    return base_math::template add_sub_limbs<NLIMBS, true, CARRY_OUT>(xs, ys, rs);
  }

  friend HOST_DEVICE IntegerRing operator+(const IntegerRing& xs, const IntegerRing& ys)
  {
    IntegerRing rs = {};
    add_limbs<TLC, false>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    return sub_modulus<1>(rs);
  }

  friend HOST_DEVICE IntegerRing operator-(const IntegerRing& xs, const IntegerRing& ys)
  {
    IntegerRing rs = {};
    uint32_t carry = sub_limbs<TLC, true>(xs.limbs_storage, ys.limbs_storage, rs.limbs_storage);
    if (carry == 0) return rs;
    const ff_storage modulus = get_modulus<1>();
    add_limbs<TLC, false>(rs.limbs_storage, modulus, rs.limbs_storage);
    return rs;
  }

  friend HOST_DEVICE bool operator==(const IntegerRing& xs, const IntegerRing& ys)
  {
    return base_math::template is_equal<TLC>(xs.limbs_storage, ys.limbs_storage);
  }

  friend std::ostream& operator<<(std::ostream& os, const IntegerRing& xs)
  {
    std::stringstream hex_string;
    hex_string << std::hex << std::setfill('0');

    for (int i = 0; i < TLC; i++) {
      hex_string << std::setw(8) << xs.limbs_storage.limbs[TLC - i - 1];
    }

    os << "0x" << hex_string.str();
    return os;
  }
};
