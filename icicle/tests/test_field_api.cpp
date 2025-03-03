
#include "test_mod_arithmetic_api.h"

// Derive all ModArith tests and add ring specific tests here
template <typename T>
class FieldTest : public ModArithTest<T>
{
};

TYPED_TEST_SUITE(FieldTest, FTImplementations);
using FieldTestBase = ModArithTestBase;

// Note: this is testing host arithmetic. Other tests against CPU backend should guarantee correct device arithmetic too
TYPED_TEST(FieldTest, FieldSanityTest)
{
  auto a = TypeParam::rand_host();
  auto b = TypeParam::rand_host();
  auto b_inv = TypeParam::inverse(b);
  auto a_neg = TypeParam::neg(a);
  ASSERT_EQ(a + TypeParam::zero(), a);
  ASSERT_EQ(a + b - a, b);
  ASSERT_EQ(b * a * b_inv, a);
  ASSERT_EQ(a + a_neg, TypeParam::zero());
  ASSERT_EQ(a * TypeParam::zero(), TypeParam::zero());
  ASSERT_EQ(b * b_inv, TypeParam::one());
  ASSERT_EQ(a * scalar_t::from(2), a + a);
}

TYPED_TEST(FieldTest, vectorDivision)
{
  const uint64_t N = 1 << rand_uint_32b(3, 17);
  const int batch_size = 1 << rand_uint_32b(0, 4);
  const bool columns_batch = rand_uint_32b(0, 1);
  const int total_size = N * batch_size;

  auto in_a = std::make_unique<TypeParam[]>(total_size);
  auto in_b = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);

  auto run = [&](const std::string& dev_type, TypeParam* out, bool measure, auto vec_op_func, const char* msg) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(divide)
    ICICLE_CHECK(vec_op_func(in_a.get(), in_b.get(), N, config, out));
    END_TIMER(divide, oss.str().c_str(), measure);
  };

  TypeParam::rand_host_many(in_a.get(), total_size);
  TypeParam::rand_host_many(in_b.get(), total_size);

  run(IcicleTestBase::reference_device(), out_ref.get(), VERBOSE, vector_div<TypeParam>, "vector div");
  run(IcicleTestBase::main_device(), out_main.get(), VERBOSE, vector_div<TypeParam>, "vector div");
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
}

TEST_F(FieldTestBase, polynomialDivision)
{
  const uint64_t numerator_size = 1 << rand_uint_32b(5, 7);
  const uint64_t denominator_size = 1 << rand_uint_32b(3, 4);
  const uint64_t q_size = numerator_size - denominator_size + 1;
  const uint64_t r_size = numerator_size;
  // const int batch_size = rand_uint_32b(10, 19);
  const int batch_size = 1; // Metal backend doesn't support batch vecops, TODO remove when supported

  // basically we compute q(x),r(x) for a(x)=q(x)b(x)+r(x) by dividing a(x)/b(x)

  auto numerator = std::make_unique<scalar_t[]>(numerator_size * batch_size);
  auto denominator = std::make_unique<scalar_t[]>(denominator_size * batch_size);

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    for (int columns_batch = 0; columns_batch <= 1; columns_batch++) {
      ICICLE_LOG_INFO << "testing polynomial division on device " << device << " [column_batch=" << columns_batch
                      << "]";

      // randomize matrix with rows/cols as polynomials
      scalar_t::rand_host_many(numerator.get(), numerator_size * batch_size);
      scalar_t::rand_host_many(denominator.get(), denominator_size * batch_size);

      // Add padding to each vector so that the degree is lower than the size
      const int zero_pad_length = 1;
      if (columns_batch) {
        for (int i = 0; i < batch_size * zero_pad_length; i++) {
          numerator[batch_size * numerator_size - batch_size * zero_pad_length + i] = scalar_t::zero();
          denominator[batch_size * denominator_size - batch_size * zero_pad_length + i] = scalar_t::zero();
        }
      } else {
        for (int i = 0; i < batch_size; ++i) {
          for (int j = 0; j < zero_pad_length; ++j) {
            numerator[i * numerator_size + numerator_size - zero_pad_length + j] = scalar_t::zero();
            denominator[i * denominator_size + denominator_size - zero_pad_length + j] = scalar_t::zero();
          }
        }
      }

      auto q = std::make_unique<scalar_t[]>(q_size * batch_size);
      auto r = std::make_unique<scalar_t[]>(r_size * batch_size);

      auto config = default_vec_ops_config();
      config.batch_size = batch_size;
      config.columns_batch = columns_batch;
      // TODO v3.2 support column batch for this API
      if (columns_batch && device == "CUDA") {
        ICICLE_LOG_INFO << "Skipping polynomial division column batch";
        continue;
      }

      ICICLE_CHECK(polynomial_division(
        numerator.get(), numerator_size, denominator.get(), denominator_size, config, q.get(), q_size, r.get(),
        r_size));

      // test a(x)=q(x)b(x)+r(x) in random point
      const auto rand_x = scalar_t::rand_host();
      auto ax = std::make_unique<scalar_t[]>(config.batch_size);
      auto bx = std::make_unique<scalar_t[]>(config.batch_size);
      auto qx = std::make_unique<scalar_t[]>(config.batch_size);
      auto rx = std::make_unique<scalar_t[]>(config.batch_size);
      polynomial_eval(numerator.get(), numerator_size, &rand_x, 1, config, ax.get());
      polynomial_eval(denominator.get(), denominator_size, &rand_x, 1, config, bx.get());
      polynomial_eval(q.get(), q_size, &rand_x, 1, config, qx.get());
      polynomial_eval(r.get(), r_size, &rand_x, 1, config, rx.get());

      for (int i = 0; i < config.batch_size; ++i) {
        ASSERT_EQ(ax[i], qx[i] * bx[i] + rx[i]);
      }
    }
  }
}

#ifdef SUMCHECK
TEST_F(FieldTestBase, Sumcheck)
{
  int log_mle_poly_size = 13;
  int mle_poly_size = 1 << log_mle_poly_size;
  int nof_mle_poly = 4;

  // generate inputs
  std::vector<scalar_t*> mle_polynomials(nof_mle_poly);
  for (int poly_i = 0; poly_i < nof_mle_poly; poly_i++) {
    mle_polynomials[poly_i] = new scalar_t[mle_poly_size];
    scalar_t::rand_host_many(mle_polynomials[poly_i], mle_poly_size);
  }

  // calculate the claimed sum
  scalar_t claimed_sum = scalar_t::zero();
  for (int element_i = 0; element_i < mle_poly_size; element_i++) {
    const scalar_t a = mle_polynomials[0][element_i];
    const scalar_t b = mle_polynomials[1][element_i];
    const scalar_t c = mle_polynomials[2][element_i];
    const scalar_t eq = mle_polynomials[3][element_i];
    claimed_sum = claimed_sum + (a * b - c) * eq;
  }

  auto run = [&](
               const std::string& dev_type, std::vector<scalar_t*>& mle_polynomials, const int mle_poly_size,
               const scalar_t claimed_sum, const char* msg) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> transcript_config; // default configuration

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // ===== Prover side ======
    // create sumcheck
    auto prover_sumcheck = create_sumcheck<scalar_t>();

    CombineFunction<scalar_t> combine_func(EQ_X_AB_MINUS_C);
    SumcheckConfig sumcheck_config;
    SumcheckProof<scalar_t> sumcheck_proof;

    START_TIMER(sumcheck);
    ICICLE_CHECK(prover_sumcheck.get_proof(
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config), sumcheck_config,
      sumcheck_proof));
    END_TIMER(sumcheck, oss.str().c_str(), true);

    // ===== Verifier side ======
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config), verification_pass));

    ASSERT_EQ(true, verification_pass);
  };

  for (const auto& device : s_registered_devices)
    run(device, mle_polynomials, mle_poly_size, claimed_sum, "Sumcheck");

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

TEST_F(FieldTestBase, SumcheckDataOnDevice)
{
  int log_mle_poly_size = 13;
  int mle_poly_size = 1 << log_mle_poly_size;
  int nof_mle_poly = 4;

  // generate inputs
  std::vector<scalar_t*> mle_polynomials(nof_mle_poly);
  for (int poly_i = 0; poly_i < nof_mle_poly; poly_i++) {
    mle_polynomials[poly_i] = new scalar_t[mle_poly_size];
    scalar_t::rand_host_many(mle_polynomials[poly_i], mle_poly_size);
  }

  // calculate the claimed sum
  scalar_t claimed_sum = scalar_t::zero();
  for (int element_i = 0; element_i < mle_poly_size; element_i++) {
    const scalar_t a = mle_polynomials[0][element_i];
    const scalar_t b = mle_polynomials[1][element_i];
    const scalar_t c = mle_polynomials[2][element_i];
    const scalar_t eq = mle_polynomials[3][element_i];
    claimed_sum = claimed_sum + (a * b - c) * eq;
  }

  std::vector<scalar_t*> data_main = std::vector<scalar_t*>(nof_mle_poly);
  icicle_set_device(IcicleTestBase::main_device());

  // create transcript_config
  SumcheckTranscriptConfig<scalar_t> transcript_config; // default configuration

  // ===== Prover side ======
  // create sumcheck
  auto prover_sumcheck = create_sumcheck<scalar_t>();

  CombineFunction<scalar_t> combine_func(EQ_X_AB_MINUS_C);
  SumcheckConfig sumcheck_config;

  sumcheck_config.are_inputs_on_device = true;

  for (int idx = 0; idx < nof_mle_poly; ++idx) {
    scalar_t* tmp = nullptr;
    icicle_malloc((void**)&tmp, mle_poly_size * sizeof(scalar_t));
    icicle_copy_to_device(tmp, mle_polynomials[idx], mle_poly_size * sizeof(scalar_t));
    data_main[idx] = tmp;
  }
  std::ostringstream oss;
  oss << IcicleTestBase::main_device() << " " << "Sumcheck";

  SumcheckProof<scalar_t> sumcheck_proof;

  START_TIMER(sumcheck);
  ICICLE_CHECK(prover_sumcheck.get_proof(
    data_main, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config), sumcheck_config,
    sumcheck_proof));
  END_TIMER(sumcheck, oss.str().c_str(), true);

  // ===== Verifier side ======
  // create sumcheck
  auto verifier_sumcheck = create_sumcheck<scalar_t>();
  bool verification_pass = false;
  ICICLE_CHECK(verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config), verification_pass));

  ASSERT_EQ(true, verification_pass);

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

MlePoly user_defined_combine(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  const MlePoly& D = inputs[3];
  return A * B - MlePoly(scalar_t::from(2)) * C + D;
}

TEST_F(FieldTestBase, SumcheckUserDefinedCombine)
{
  int log_mle_poly_size = 13;
  int mle_poly_size = 1 << log_mle_poly_size;
  int nof_mle_poly = 4;

  // generate inputs
  std::vector<scalar_t*> mle_polynomials(nof_mle_poly);
  for (int poly_i = 0; poly_i < nof_mle_poly; poly_i++) {
    mle_polynomials[poly_i] = new scalar_t[mle_poly_size];
    scalar_t::rand_host_many(mle_polynomials[poly_i], mle_poly_size);
  }

  // calculate the claimed sum
  scalar_t claimed_sum = scalar_t::zero();
  for (int element_i = 0; element_i < mle_poly_size; element_i++) {
    const scalar_t a = mle_polynomials[0][element_i];
    const scalar_t b = mle_polynomials[1][element_i];
    const scalar_t c = mle_polynomials[2][element_i];
    const scalar_t d = mle_polynomials[3][element_i];
    claimed_sum = claimed_sum + (a * b - scalar_t::from(2) * c + d);
  }

  auto run = [&](
               const std::string& dev_type, std::vector<scalar_t*>& mle_polynomials, const int mle_poly_size,
               const scalar_t claimed_sum, const char* msg) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> transcript_config; // default configuration

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // ===== Prover side ======
    // create sumcheck
    auto prover_sumcheck = create_sumcheck<scalar_t>();

    CombineFunction<scalar_t> combine_func(user_defined_combine, nof_mle_poly);
    SumcheckConfig sumcheck_config;
    SumcheckProof<scalar_t> sumcheck_proof;

    START_TIMER(sumcheck);
    ICICLE_CHECK(prover_sumcheck.get_proof(
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config), sumcheck_config,
      sumcheck_proof));
    END_TIMER(sumcheck, oss.str().c_str(), true);

    // ===== Verifier side ======
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config), verification_pass));

    ASSERT_EQ(true, verification_pass);
  };
  for (const auto& device : s_registered_devices) {
    run(device, mle_polynomials, mle_poly_size, claimed_sum, "Sumcheck");
  }

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

  #ifdef CUDA_ARCH

MlePoly too_complex_combine(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  return A * B + B * C + C * A + A * B * C - scalar_t::from(2) + scalar_t::from(9) + A * B * C + C * B * A;
}

MlePoly too_high_degree_combine(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  return (A * B * C * A * B * C * A * B * C);
}

MlePoly too_many_polynomials_combine(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  const MlePoly& D = inputs[3];
  const MlePoly& E = inputs[4];
  const MlePoly& F = inputs[5];
  const MlePoly& G = inputs[5];
  const MlePoly& H = inputs[5];
  const MlePoly& I = inputs[5];
  return A * B * C + D * E * F + G * H * I;
}

TEST_F(FieldTestBase, SumcheckCudaShouldFailCases)
{
  int log_mle_poly_size = 13;
  int mle_poly_size = 1 << log_mle_poly_size;
  int nof_mle_poly_big = 9;
  int nof_mle_poly = 6;
  int nof_mle_poly_small = 3;

  // generate inputs
  std::vector<scalar_t*> mle_polynomials_big(nof_mle_poly_big);
  for (int poly_i = 0; poly_i < nof_mle_poly_big; poly_i++) {
    mle_polynomials_big[poly_i] = new scalar_t[mle_poly_size];
    scalar_t::rand_host_many(mle_polynomials_big[poly_i], mle_poly_size);
  }

  std::vector<scalar_t*> mle_polynomials(nof_mle_poly);
  for (int poly_i = 0; poly_i < nof_mle_poly; poly_i++) {
    mle_polynomials[poly_i] = new scalar_t[mle_poly_size];
    scalar_t::rand_host_many(mle_polynomials[poly_i], mle_poly_size);
  }

  std::vector<scalar_t*> mle_polynomials_small(nof_mle_poly_small);
  for (int poly_i = 0; poly_i < nof_mle_poly_small; poly_i++) {
    mle_polynomials_small[poly_i] = new scalar_t[mle_poly_size];
    scalar_t::rand_host_many(mle_polynomials_small[poly_i], mle_poly_size);
  }

  // claimed sum
  scalar_t claimed_sum = scalar_t::zero();

  auto run = [&](
               const std::string& dev_type, std::vector<scalar_t*>& mle_polynomials, const int mle_poly_size,
               const scalar_t claimed_sum, CombineFunction<scalar_t> combine_func) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> transcript_config; // default configuration

    // ===== Prover side ======
    // create sumcheck
    auto prover_sumcheck = create_sumcheck<scalar_t>();
    SumcheckConfig sumcheck_config;
    SumcheckProof<scalar_t> sumcheck_proof;

    eIcicleError error = prover_sumcheck.get_proof(
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config), sumcheck_config,
      sumcheck_proof);

    ASSERT_EQ(error, eIcicleError::INVALID_ARGUMENT);
  };

  CombineFunction<scalar_t> combine_func_too_many_polys(too_many_polynomials_combine, nof_mle_poly_big);
  run("CUDA", mle_polynomials_big, mle_poly_size, claimed_sum, combine_func_too_many_polys);
  CombineFunction<scalar_t> combine_func_too_complex(too_complex_combine, nof_mle_poly_small);
  run("CUDA", mle_polynomials_small, mle_poly_size, claimed_sum, combine_func_too_complex);
  CombineFunction<scalar_t> combine_func_too_high_degree(too_high_degree_combine, nof_mle_poly_small);
  run("CUDA", mle_polynomials_small, mle_poly_size, claimed_sum, combine_func_too_high_degree);

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
  for (auto& mle_poly_ptr : mle_polynomials_small) {
    delete[] mle_poly_ptr;
  }
}
  #endif // CUDA_ARCH

MlePoly identity(const std::vector<MlePoly>& inputs) { return inputs[0]; }

TEST_F(FieldTestBase, SumcheckIdentity)
{
  int log_mle_poly_size = 13;
  int mle_poly_size = 1 << log_mle_poly_size;
  int nof_mle_poly = 1;

  // generate inputs
  std::vector<scalar_t*> mle_polynomials(nof_mle_poly);
  for (int poly_i = 0; poly_i < nof_mle_poly; poly_i++) {
    mle_polynomials[poly_i] = new scalar_t[mle_poly_size];
    scalar_t::rand_host_many(mle_polynomials[poly_i], mle_poly_size);
  }

  // calculate the claimed sum
  scalar_t claimed_sum = scalar_t::zero();
  for (int element_i = 0; element_i < mle_poly_size; element_i++) {
    const scalar_t a = mle_polynomials[0][element_i];
    claimed_sum = claimed_sum + a;
  }

  auto run = [&](
               const std::string& dev_type, std::vector<scalar_t*>& mle_polynomials, const int mle_poly_size,
               const scalar_t claimed_sum, const char* msg) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> transcript_config; // default configuration

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // ===== Prover side ======
    // create sumcheck
    auto prover_sumcheck = create_sumcheck<scalar_t>();

    CombineFunction<scalar_t> combine_func(identity, nof_mle_poly);
    SumcheckConfig sumcheck_config;
    SumcheckProof<scalar_t> sumcheck_proof;

    START_TIMER(sumcheck);
    ICICLE_CHECK(prover_sumcheck.get_proof(
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config), sumcheck_config,
      sumcheck_proof));
    END_TIMER(sumcheck, oss.str().c_str(), true);

    // ===== Verifier side ======
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config), verification_pass));

    ASSERT_EQ(true, verification_pass);
  };

  for (const auto& device : s_registered_devices)
    run(device, mle_polynomials, mle_poly_size, claimed_sum, "Sumcheck");

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

MlePoly single_input(const std::vector<MlePoly>& inputs) { return MlePoly(scalar_t::from(2)) * inputs[0]; }

TEST_F(FieldTestBase, SumcheckSingleInputProgram)
{
  int log_mle_poly_size = 13;
  int mle_poly_size = 1 << log_mle_poly_size;
  int nof_mle_poly = 1;

  // generate inputs
  std::vector<scalar_t*> mle_polynomials(nof_mle_poly);
  for (int poly_i = 0; poly_i < nof_mle_poly; poly_i++) {
    mle_polynomials[poly_i] = new scalar_t[mle_poly_size];
    scalar_t::rand_host_many(mle_polynomials[poly_i], mle_poly_size);
  }

  // calculate the claimed sum
  scalar_t claimed_sum = scalar_t::zero();
  for (int element_i = 0; element_i < mle_poly_size; element_i++) {
    const scalar_t a = mle_polynomials[0][element_i];
    claimed_sum = claimed_sum + scalar_t::from(2) * a;
  }

  auto run = [&](
               const std::string& dev_type, std::vector<scalar_t*>& mle_polynomials, const int mle_poly_size,
               const scalar_t claimed_sum, const char* msg) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> transcript_config; // default configuration

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // ===== Prover side ======
    // create sumcheck
    auto prover_sumcheck = create_sumcheck<scalar_t>();

    CombineFunction<scalar_t> combine_func(single_input, nof_mle_poly);
    SumcheckConfig sumcheck_config;
    SumcheckProof<scalar_t> sumcheck_proof;

    START_TIMER(sumcheck);
    ICICLE_CHECK(prover_sumcheck.get_proof(
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config), sumcheck_config,
      sumcheck_proof));
    END_TIMER(sumcheck, oss.str().c_str(), true);

    // ===== Verifier side ======
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config), verification_pass));

    ASSERT_EQ(true, verification_pass);
  };

  for (const auto& device : s_registered_devices)
    run(device, mle_polynomials, mle_poly_size, claimed_sum, "Sumcheck");

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

#endif // SUMCHECK

// TODO Hadar: this is a workaround for 'storage<18 - scalar_t::TLC>' failing due to 17 limbs not supported.
//             It means we skip fields such as babybear!
// TODO: this test make problem for curves too as they have extension fields too. Need to clean it up TODO Hadar

// #ifndef EXT_FIELD
// TEST_F(FieldTestBase, FieldStorageReduceSanityTest)
// {
//   /*
//   SR - storage reduce
//   check that:
//   1. SR(x1) + SR(x1) = SR(x1+x2)
//   2. SR(INV(SR(x))*x) = 1
//   */
//   START_TIMER(StorageSanity)
//   for (int i = 0; i < 1000; i++) {
//     if constexpr (scalar_t::TLC == 1) {
//       storage<18> a =                                          // 18 because we support up to 576 bits
//         scalar_t::template rand_storage<18>(17);               // 17 so we don't have carry after addition
//       storage<18> b = scalar_t::template rand_storage<18>(17); // 17 so we don't have carry after addition
//       storage<18> sum = {};
//       const storage<3> c =
//         scalar_t::template rand_storage<3>(); // 3 because we don't support higher odd number of limbs yet
//       storage<4> product = {};
//       host_math::template add_sub_limbs<18, false, false, true>(a, b, sum);
//       auto c_red = scalar_t::from(c);
//       auto c_inv = scalar_t::inverse(c_red);
//       host_math::multiply_raw<3, 1, true>(
//         c, c_inv.limbs_storage, product); // using 32-bit multiplication for small fields
//       ASSERT_EQ(scalar_t::from(a) + scalar_t::from(b), scalar_t::from(sum));
//       ASSERT_EQ(scalar_t::from(product), scalar_t::one());
//       std::byte* a_bytes = reinterpret_cast<std::byte*>(a.limbs);
//       std::byte* b_bytes = reinterpret_cast<std::byte*>(b.limbs);
//       std::byte* sum_bytes = reinterpret_cast<std::byte*>(sum.limbs);
//       std::byte* product_bytes = reinterpret_cast<std::byte*>(product.limbs);
//       ASSERT_EQ(scalar_t::from(a), scalar_t::from(a_bytes, 18 * 4));
//       ASSERT_EQ(scalar_t::from(a_bytes, 18 * 4) + scalar_t::from(b_bytes, 18 * 4), scalar_t::from(sum_bytes, 18 *
//       4)); ASSERT_EQ(scalar_t::from(product_bytes, 4 * 4), scalar_t::one());
//     } else {
//       storage<18> a =                                          // 18 because we support up to 576 bits
//         scalar_t::template rand_storage<18>(17);               // 17 so we don't have carry after addition
//       storage<18> b = scalar_t::template rand_storage<18>(17); // 17 so we don't have carry after addition
//       storage<18> sum = {};
//       const storage<18 - scalar_t::TLC> c =
//         scalar_t::template rand_storage<18 - scalar_t::TLC>(); // -TLC so we don't overflow in multiplication
//       storage<18> product = {};
//       host_math::template add_sub_limbs<18, false, false, true>(a, b, sum);
//       auto c_red = scalar_t::from(c);
//       auto c_inv = scalar_t::inverse(c_red);
//       host_math::multiply_raw(c, c_inv.limbs_storage, product);
//       ASSERT_EQ(scalar_t::from(a) + scalar_t::from(b), scalar_t::from(sum));
//       ASSERT_EQ(scalar_t::from(product), scalar_t::one());
//       std::byte* a_bytes = reinterpret_cast<std::byte*>(a.limbs);
//       std::byte* b_bytes = reinterpret_cast<std::byte*>(b.limbs);
//       std::byte* sum_bytes = reinterpret_cast<std::byte*>(sum.limbs);
//       std::byte* product_bytes = reinterpret_cast<std::byte*>(product.limbs);
//       ASSERT_EQ(scalar_t::from(a), scalar_t::from(a_bytes, 18 * 4));
//       ASSERT_EQ(scalar_t::from(a_bytes, 18 * 4) + scalar_t::from(b_bytes, 18 * 4), scalar_t::from(sum_bytes, 18 *
//       4)); ASSERT_EQ(scalar_t::from(product_bytes, 18 * 4), scalar_t::one());
//     }
//   }
//   END_TIMER(StorageSanity, "storage sanity", true);
// }
// #endif // ! EXT_FIELD