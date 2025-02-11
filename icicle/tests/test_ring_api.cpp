#include "test_base.h"

// TODO Yuval: control the exact type of Zq by the build system
#include "icicle/rings/integer_rings/labrador.h"

class RingApiTestBase : public IcicleTestBase
{
};

TEST_F(RingApiTestBase, RingSanityTest)
{
  using namespace labrador;
  ICICLE_LOG_INFO << "Hello from RingSanityTest";
  ICICLE_LOG_INFO << "omegas_count = " << labrador::zq_config::omegas_count;

  // auto a = Zq::rand_host();
  // //   auto b = Zq::rand_host();
  // // auto b = Zq::from(0x78000001); // stuck inverse due to no inverse!
  // auto b_inv = Zq::inverse(b);
  // auto a_neg = Zq::neg(a);
  // ASSERT_EQ(a + Zq::zero(), a);
  // ASSERT_EQ(a + b - a, b);
  // ASSERT_EQ(b * a * b_inv, a);
  // ASSERT_EQ(a + a_neg, Zq::zero());
  // ASSERT_EQ(a * Zq::zero(), Zq::zero());
  // ASSERT_EQ(b * b_inv, Zq::one());
  // ASSERT_EQ(a * Zq::from(2), a + a);

  constexpr int logn = 27;
  auto w = Zq::omega(logn);
  ICICLE_LOG_INFO << "omega(" << logn << ") = " << w;
  ICICLE_LOG_INFO << Zq::pow(w, 1 << (logn));
}