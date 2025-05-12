#include "test_mod_arithmetic_api.h"
#include "icicle/sumcheck/sumcheck.h"
#include "icicle/fri/fri.h"
#include "icicle/fri/fri_config.h"
#include "icicle/fri/fri_proof.h"
#include "icicle/fri/fri_transcript_config.h"
#include "icicle/sumcheck/sumcheck_proof_serializer.h"
#include "icicle/fri/fri_proof_serializer.h"

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
  ASSERT_EQ(TypeParam::from_montgomery(TypeParam::to_montgomery(a)), a);
  ASSERT_EQ(a + TypeParam::zero(), a);
  ASSERT_EQ(a + b - a, b);
  ASSERT_EQ(b * a * b_inv, a);
  ASSERT_EQ(a + a_neg, TypeParam::zero());
  ASSERT_EQ(a * TypeParam::zero(), TypeParam::zero());
  ASSERT_EQ(b * b_inv, TypeParam::one());
  ASSERT_EQ(a * scalar_t::from(2), a + a);
}

TYPED_TEST(FieldTest, NTTTest)
{
  const uint64_t N = 1 << 3; // rand_uint_32b(3, 17);
  const int batch_size = 1 << 0; // rand_uint_32b(0, 4);
  const bool columns_batch = false; // rand_uint_32b(0, 1);
  const NTTDir dir = NTTDir::kForward; // static_cast<NTTDir>(rand_uint_32b(0, 1));
  const bool inplace = false; // rand_uint_32b(0, 1);

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;
  ICICLE_LOG_DEBUG << "dir = " << (dir == NTTDir::kForward ? "forward" : "inverse");
  ICICLE_LOG_DEBUG << "inplace = " << inplace;

  const int total_size = N * batch_size;
  auto in_a = std::make_unique<TypeParam[]>(total_size);
  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);

  // Initialize input data with alternating 0s and 1s
  for (int i = 0; i < total_size; i++) {
    in_a[i] = (i % 2 == 0) ? TypeParam::zero() : TypeParam::one();
  }
  
  // Print in_a values for debugging
  ICICLE_LOG_INFO << "in_a values after initialization:";
  for (int i = 0; i < total_size; i++) {
    ICICLE_LOG_INFO << "in_a[" << i << "] = " << in_a[i];
  }

  auto run = [&](const std::string& dev_type, TypeParam* out, bool measure, const char* msg, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // Initialize NTT domain
    auto init_domain_config = default_ntt_init_domain_config();
    ICICLE_CHECK(ntt_init_domain(scalar_t::omega(log2(N)), init_domain_config));

    auto config = default_ntt_config<TypeParam>();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;

    // Print NTT configuration
    ICICLE_LOG_INFO << "NTT Configuration for " << dev_type << ":";
    ICICLE_LOG_INFO << "  - Size: " << N;
    ICICLE_LOG_INFO << "  - Batch size: " << config.batch_size;
    ICICLE_LOG_INFO << "  - Columns batch: " << config.columns_batch;
    ICICLE_LOG_INFO << "  - Direction: " << (dir == NTTDir::kForward ? "Forward" : "Inverse");
    ICICLE_LOG_INFO << "  - Ordering: " << static_cast<int>(config.ordering);
    ICICLE_LOG_INFO << "  - Inputs on device: " << config.are_inputs_on_device;
    ICICLE_LOG_INFO << "  - Outputs on device: " << config.are_outputs_on_device;
    ICICLE_LOG_INFO << "  - Is async: " << config.is_async;

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(NTT_sync)
    for (int i = 0; i < iters; ++i) {
      ICICLE_CHECK(ntt(in_a.get(), N, dir, config, out)); //TODO: ..., inplace ? in_a.get() : out));
    }
    END_TIMER(NTT_sync, oss.str().c_str(), measure);

    // Release NTT domain
    ICICLE_CHECK(ntt_release_domain<scalar_t>());
  };

  // Run reference implementation (CPU)
  run(IcicleTestBase::reference_device(), out_ref.get(), VERBOSE, "NTT", ITERS);

  // Print out_ref values after CPU NTT
  ICICLE_LOG_INFO << "out_ref values after CPU NTT:";
  for (int i = 0; i < total_size; i++) {
    ICICLE_LOG_INFO << "out_ref[" << i << "] = " << out_ref[i];
  }

  // // Run main implementation
  // run(IcicleTestBase::main_device(), out_main.get(), VERBOSE, "NTT", ITERS);

  // // Compare results
  // if (inplace) {
  //   ASSERT_EQ(0, memcmp(in_a.get(), out_ref.get(), total_size * sizeof(TypeParam)));
  // } else {
  //   ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(TypeParam)));
  // }

  // Run Vulkan implementation if available
  if (std::find(FieldTest<TypeParam>::s_registered_devices.begin(), FieldTest<TypeParam>::s_registered_devices.end(), "VULKAN") != FieldTest<TypeParam>::s_registered_devices.end()) {
    auto out_vulkan = std::make_unique<TypeParam[]>(total_size);
    
    // Initialize NTT domain for Vulkan
    Device vulkan_dev = {"VULKAN", 0};
    icicle_set_device(vulkan_dev);

    // Print input data before Vulkan NTT
    ICICLE_LOG_INFO << "Input data before Vulkan NTT:";
    for (int i = 0; i < total_size; i++) {
      ICICLE_LOG_INFO << "in_a[" << i << "] = " << in_a[i];
    }

    // Allocate device memory and copy input data
    TypeParam* d_in = nullptr;
    TypeParam* d_out = nullptr;
    ICICLE_CHECK(icicle_malloc((void**)&d_in, total_size * sizeof(TypeParam)));
    ICICLE_CHECK(icicle_malloc((void**)&d_out, total_size * sizeof(TypeParam)));
    ICICLE_CHECK(icicle_copy_to_device(d_in, in_a.get(), total_size * sizeof(TypeParam)));

    auto init_domain_config = default_ntt_init_domain_config();
    ICICLE_LOG_INFO << "before init domain";
    ICICLE_CHECK(ntt_init_domain(scalar_t::omega(log2(N)), init_domain_config));
    ICICLE_LOG_INFO << "after init domain";

    auto config = default_ntt_config<TypeParam>();
    config.batch_size = batch_size;
    config.columns_batch = columns_batch;
    config.are_inputs_on_device = true;
    config.are_outputs_on_device = true;

    ICICLE_LOG_INFO << "NTT Configuration for Vulkan:";
    ICICLE_LOG_INFO << "  - Size: " << N;
    ICICLE_LOG_INFO << "  - Batch size: " << config.batch_size;
    ICICLE_LOG_INFO << "  - Columns batch: " << config.columns_batch;
    ICICLE_LOG_INFO << "  - Direction: " << (dir == NTTDir::kForward ? "Forward" : "Inverse");
    ICICLE_LOG_INFO << "  - Ordering: " << static_cast<int>(config.ordering);
    ICICLE_LOG_INFO << "  - Inputs on device: " << config.are_inputs_on_device;
    ICICLE_LOG_INFO << "  - Outputs on device: " << config.are_outputs_on_device;
    ICICLE_LOG_INFO << "  - Is async: " << config.is_async;


    ICICLE_LOG_INFO << "before ntt";

    START_TIMER(NTT_sync)
    ICICLE_CHECK(ntt(d_in, N, dir, config, inplace ? d_in : d_out));
    END_TIMER(NTT_sync, "VULKAN NTT", VERBOSE);

    // Copy result back to host
    ICICLE_CHECK(icicle_copy_to_host(out_vulkan.get(), inplace ? d_in : d_out, total_size * sizeof(TypeParam)));

    // Free device memory
    ICICLE_CHECK(icicle_free(d_in));
    ICICLE_CHECK(icicle_free(d_out));

    // Release NTT domain for Vulkan
    ICICLE_CHECK(ntt_release_domain<scalar_t>());
    
    // Print out_vulkan values before comparison
    ICICLE_LOG_INFO << "out_vulkan values before comparison:";
    for (int i = 0; i < total_size; i++) {
      ICICLE_LOG_INFO << "out_vulkan[" << i << "] = " << out_vulkan[i];
    }
    
    // Compare Vulkan results with reference
    if (inplace) {
      ASSERT_EQ(0, memcmp(in_a.get(), out_ref.get(), total_size * sizeof(TypeParam)));
    } else {
      ASSERT_EQ(0, memcmp(out_vulkan.get(), out_ref.get(), total_size * sizeof(TypeParam)));
    }
  }
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
  const int batch_size = rand_uint_32b(10, 19);

  // basically we compute q(x),r(x) for a(x)=q(x)b(x)+r(x) by dividing a(x)/b(x)

  auto numerator = std::make_unique<scalar_t[]>(numerator_size * batch_size);
  auto denominator = std::make_unique<scalar_t[]>(denominator_size * batch_size);

  for (auto device : s_registered_devices) {
    ICICLE_CHECK(icicle_set_device(device));
    for (int columns_batch = 0; columns_batch <= 1; columns_batch++) {
      // TODO @Hadar support column batch for this API
      if (columns_batch && (device == "CUDA" || device == "METAL")) {
        ICICLE_LOG_INFO << "Skipping polynomial division column batch";
        continue;
      }

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
  #include "icicle/hash/keccak.h"
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

    // ===== Prover side ======

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> transcript_config(
      create_keccak_256_hash(), "labelA", "labelB", "LabelC", scalar_t::from(12));

    ASSERT_NE(transcript_config.get_domain_separator_label().size(),
              0); // assert label exists

    std::ostringstream oss;
    oss << dev_type << " " << msg;

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

    ASSERT_EQ(transcript_config.get_domain_separator_label().size(), 0); // assert data was moved and not copied

    // ===== Verifier side ======
    // Note that the verifier is another machine and needs to regenerate the same transcript config.
    // Also note that even if the same process, the transcript-config is moved since it may be large, so cannot reuse
    // twice.
    SumcheckTranscriptConfig<scalar_t> verifier_transcript_config(
      create_keccak_256_hash(), "labelA", "labelB", "LabelC", scalar_t::from(12));
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(verifier_transcript_config), verification_pass));

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
  SumcheckTranscriptConfig<scalar_t> transcript_config_prover;   // default configuration
  SumcheckTranscriptConfig<scalar_t> transcript_config_verifier; // default configuration

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
  oss << IcicleTestBase::main_device() << " "
      << "Sumcheck";

  SumcheckProof<scalar_t> sumcheck_proof;

  START_TIMER(sumcheck);
  ICICLE_CHECK(prover_sumcheck.get_proof(
    data_main, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config_prover), sumcheck_config,
    sumcheck_proof));
  END_TIMER(sumcheck, oss.str().c_str(), true);

  // ===== Verifier side ======
  SumcheckTranscriptConfig<scalar_t> verifier_transcript_config; // default configuration
  // create sumcheck
  auto verifier_sumcheck = create_sumcheck<scalar_t>();
  bool verification_pass = false;
  ICICLE_CHECK(
    verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(verifier_transcript_config), verification_pass));

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
    SumcheckTranscriptConfig<scalar_t> verifier_transcript_config; // default configuration
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(verifier_transcript_config), verification_pass));

    ASSERT_EQ(true, verification_pass);
  };
  for (const auto& device : s_registered_devices) {
    run(device, mle_polynomials, mle_poly_size, claimed_sum, "Sumcheck");
  }

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

MlePoly max_allowed_degree_combine(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  const MlePoly& D = inputs[3];
  const MlePoly& E = inputs[4];
  const MlePoly& F = inputs[5];
  return A * B * C * D * E * F;
}

TEST_F(FieldTestBase, SumcheckMaxAllowedDegreeCombine)
{
  int log_mle_poly_size = 13;
  int mle_poly_size = 1 << log_mle_poly_size;
  int nof_mle_poly = 6;

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
    const scalar_t e = mle_polynomials[4][element_i];
    const scalar_t f = mle_polynomials[5][element_i];
    claimed_sum = claimed_sum + (a * b * c * d * e * f);
  }

  auto run = [&](
               const std::string& dev_type, std::vector<scalar_t*>& mle_polynomials, const int mle_poly_size,
               const scalar_t claimed_sum, const char* msg) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> transcript_config_prover;   // default configuration
    SumcheckTranscriptConfig<scalar_t> transcript_config_verifier; // default configuration

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // ===== Prover side ======
    // create sumcheck
    auto prover_sumcheck = create_sumcheck<scalar_t>();

    CombineFunction<scalar_t> combine_func(max_allowed_degree_combine, nof_mle_poly);
    SumcheckConfig sumcheck_config;
    SumcheckProof<scalar_t> sumcheck_proof;

    START_TIMER(sumcheck);
    ICICLE_CHECK(prover_sumcheck.get_proof(
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config_prover), sumcheck_config,
      sumcheck_proof));
    END_TIMER(sumcheck, oss.str().c_str(), true);

    // ===== Verifier side ======
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config_verifier), verification_pass));

    ASSERT_EQ(true, verification_pass);
  };
  for (const auto& device : s_registered_devices) {
    run(device, mle_polynomials, mle_poly_size, claimed_sum, "Sumcheck");
  }

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

MlePoly max_allowed_nof_polys_comine(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  const MlePoly& D = inputs[3];
  const MlePoly& E = inputs[4];
  const MlePoly& F = inputs[5];
  const MlePoly& G = inputs[6];
  const MlePoly& H = inputs[7];
  return A * B * C + D * E * F + G - H;
}

TEST_F(FieldTestBase, SumcheckMaxAllowedNofPolys)
{
  int log_mle_poly_size = 13;
  int mle_poly_size = 1 << log_mle_poly_size;
  int nof_mle_poly = 8;

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
    const scalar_t e = mle_polynomials[4][element_i];
    const scalar_t f = mle_polynomials[5][element_i];
    const scalar_t g = mle_polynomials[6][element_i];
    const scalar_t h = mle_polynomials[7][element_i];
    claimed_sum = claimed_sum + (a * b * c + d * e * f + g - h);
  }

  auto run = [&](
               const std::string& dev_type, std::vector<scalar_t*>& mle_polynomials, const int mle_poly_size,
               const scalar_t claimed_sum, const char* msg) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> transcript_config_prover;   // default configuration
    SumcheckTranscriptConfig<scalar_t> transcript_config_verifier; // default configuration

    std::ostringstream oss;
    oss << dev_type << " " << msg;
    // ===== Prover side ======
    // create sumcheck
    auto prover_sumcheck = create_sumcheck<scalar_t>();

    CombineFunction<scalar_t> combine_func(max_allowed_nof_polys_comine, nof_mle_poly);
    SumcheckConfig sumcheck_config;
    SumcheckProof<scalar_t> sumcheck_proof;

    START_TIMER(sumcheck);
    ICICLE_CHECK(prover_sumcheck.get_proof(
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config_prover), sumcheck_config,
      sumcheck_proof));
    END_TIMER(sumcheck, oss.str().c_str(), true);

    // ===== Verifier side ======
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config_verifier), verification_pass));

    ASSERT_EQ(true, verification_pass);
  };
  for (const auto& device : s_registered_devices) {
    run(device, mle_polynomials, mle_poly_size, claimed_sum, "Sumcheck");
  }

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

MlePoly identity(const std::vector<MlePoly>& inputs) { return inputs[0]; }

TEST_F(FieldTestBase, SumcheckDifferentTranscriptShouldFail)
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

    Hash hasher = create_keccak_256_hash();
    const char* domain_label = "ingonyama";
    const char* poly_label = "poly_label";
    const char* challenge_label = "icicle";
    scalar_t seed = scalar_t::from(18);
    bool little_endian = true;

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> transcript_config_prover(
      hasher, domain_label, poly_label, challenge_label, seed, little_endian);
    SumcheckTranscriptConfig<scalar_t> transcript_config_verifier; // default configuration

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
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config_prover), sumcheck_config,
      sumcheck_proof));
    END_TIMER(sumcheck, oss.str().c_str(), true);

    // ===== Verifier side ======
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(transcript_config_verifier), verification_pass));

    ASSERT_EQ(false, verification_pass);
  };

  for (const auto& device : s_registered_devices)
    run(device, mle_polynomials, mle_poly_size, claimed_sum, "Sumcheck");

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

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

TEST_F(FieldTestBase, SumcheckDeviceShouldFailCases)
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
  for (const auto& device : s_registered_devices) {
    if (device == "CPU") continue;
    ICICLE_LOG_INFO << "Run test on device: " << device;
    CombineFunction<scalar_t> combine_func_too_many_polys(too_many_polynomials_combine, nof_mle_poly_big);
    run(device, mle_polynomials_big, mle_poly_size, claimed_sum, combine_func_too_many_polys);
    CombineFunction<scalar_t> combine_func_too_complex(too_complex_combine, nof_mle_poly_small);
    run(device, mle_polynomials_small, mle_poly_size, claimed_sum, combine_func_too_complex);
    CombineFunction<scalar_t> combine_func_too_high_degree(too_high_degree_combine, nof_mle_poly_small);
    run(device, mle_polynomials_small, mle_poly_size, claimed_sum, combine_func_too_high_degree);
  }

  for (auto& mle_poly_ptr : mle_polynomials_big) {
    delete[] mle_poly_ptr;
  }
  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
  for (auto& mle_poly_ptr : mle_polynomials_small) {
    delete[] mle_poly_ptr;
  }
}

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

    Hash hasher = create_keccak_256_hash();
    const char* domain_label = "ingonyama";
    const char* poly_label = "poly_label";
    const char* challenge_label = "icicle";
    scalar_t seed = scalar_t::from(18);
    bool little_endian = true;

    // create transcript_config
    SumcheckTranscriptConfig<scalar_t> config_prover; // default configuration

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
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(config_prover), sumcheck_config,
      sumcheck_proof));
    END_TIMER(sumcheck, oss.str().c_str(), true);

    // ===== Verifier side ======
    SumcheckTranscriptConfig<scalar_t> verifier_transcript_config; // default configuration
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(verifier_transcript_config), verification_pass));

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
    SumcheckTranscriptConfig<scalar_t> transcript_config_prover;   // default configuration
    SumcheckTranscriptConfig<scalar_t> transcript_config_verifier; // default configuration

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
      mle_polynomials, mle_poly_size, claimed_sum, combine_func, std::move(transcript_config_prover), sumcheck_config,
      sumcheck_proof));
    END_TIMER(sumcheck, oss.str().c_str(), true);

    // ===== Verifier side ======
    SumcheckTranscriptConfig<scalar_t> verifier_transcript_config; // default configuration
    // create sumcheck
    auto verifier_sumcheck = create_sumcheck<scalar_t>();
    bool verification_pass = false;
    ICICLE_CHECK(
      verifier_sumcheck.verify(sumcheck_proof, claimed_sum, std::move(verifier_transcript_config), verification_pass));

    ASSERT_EQ(true, verification_pass);

    // Serialize proof
    size_t proof_size = 0;
    ICICLE_CHECK(BinarySerializer<SumcheckProof<scalar_t>>::serialized_size(sumcheck_proof, proof_size));
    std::vector<std::byte> proof_bytes(proof_size);
    ICICLE_CHECK(
      BinarySerializer<SumcheckProof<scalar_t>>::serialize(proof_bytes.data(), proof_bytes.size(), sumcheck_proof));

    // Deserialize proof
    SumcheckProof<scalar_t> deserialized_proof;
    ICICLE_CHECK(BinarySerializer<SumcheckProof<scalar_t>>::deserialize(
      proof_bytes.data(), proof_bytes.size(), deserialized_proof));

    // Compare proofs
    uint nof_round_polynomials = sumcheck_proof.get_nof_round_polynomials();
    ASSERT_EQ(nof_round_polynomials, deserialized_proof.get_nof_round_polynomials());
    for (uint round_i = 0; round_i < nof_round_polynomials; round_i++) {
      const auto& round_poly = sumcheck_proof.get_round_polynomial(round_i);
      const auto& deserialized_round_poly = deserialized_proof.get_round_polynomial(round_i);
      ASSERT_EQ(round_poly, deserialized_round_poly);
    }
  };

  for (const auto& device : s_registered_devices)
    run(device, mle_polynomials, mle_poly_size, claimed_sum, "Sumcheck");

  for (auto& mle_poly_ptr : mle_polynomials) {
    delete[] mle_poly_ptr;
  }
}

#endif // SUMCHECK

#ifdef FRI

TYPED_TEST(FieldTest, Fri)
{
  size_t log_stopping_size;
  size_t pow_bits;
  size_t nof_queries;
  for (size_t params_options = 0; params_options <= 1; params_options++) {
    if (params_options) {
      log_stopping_size = 0;
      pow_bits = 16;
      nof_queries = 100;
    } else {
      log_stopping_size = 8;
      pow_bits = 0;
      nof_queries = 50;
    }
    for (size_t log_input_size = 16; log_input_size <= 24; log_input_size += 4) {
      const size_t input_size = 1 << log_input_size;
      const size_t folding_factor = 2; // TODO SHANIE (future) - add support for other folding factors
      const size_t stopping_size = 1 << log_stopping_size;
      const size_t stopping_degree = stopping_size - 1;
      const uint64_t output_store_min_layer = 0;

      // Generate input polynomial evaluations
      auto scalars = std::make_unique<TypeParam[]>(input_size);
      TypeParam::rand_host_many(scalars.get(), input_size);

      auto run = [log_input_size, input_size, folding_factor, stopping_degree, output_store_min_layer, nof_queries,
                  pow_bits, &scalars](const std::string& dev_type, bool measure) {
        Device dev = {dev_type, 0};
        ICICLE_CHECK(icicle_set_device(dev));

        // Initialize ntt domain
        NTTInitDomainConfig init_domain_config = default_ntt_init_domain_config();
        ICICLE_CHECK(ntt_init_domain(scalar_t::omega(log_input_size), init_domain_config));

        // ===== Prover side ======
        uint64_t merkle_tree_arity = 2; // TODO SHANIE (future) - add support for other arities

        // Define hashers for merkle tree
        Hash hash = Keccak256::create(sizeof(TypeParam));                          // hash element -> 32B
        Hash compress = Keccak256::create(merkle_tree_arity * hash.output_size()); // hash every 64B to 32B

        // set transcript config
        const char* domain_separator_label = "domain_separator_label";
        const char* round_challenge_label = "round_challenge_label";
        const char* commit_phase_label = "commit_phase_label";
        const char* nonce_label = "nonce_label";
        std::vector<std::byte>&& public_state = {};
        TypeParam seed_rng = TypeParam::one();

        FriTranscriptConfig<TypeParam> transcript_config(
          hash, domain_separator_label, round_challenge_label, commit_phase_label, nonce_label, std::move(public_state),
          seed_rng);

        FriConfig fri_config;
        fri_config.nof_queries = nof_queries;
        fri_config.pow_bits = pow_bits;
        fri_config.folding_factor = folding_factor;
        fri_config.stopping_degree = stopping_degree;
        FriProof<TypeParam> fri_proof;

        std::ostringstream oss;
        if (measure) {
          ICICLE_LOG_INFO << "log_input_size: " << log_input_size << ". stopping_degree: " << stopping_degree
                          << ". pow_bits: " << pow_bits << ". nof_queries:" << nof_queries;
          oss << dev_type << " FRI proof";
        }
        START_TIMER(FRIPROOF_sync)
        eIcicleError err = fri_merkle_tree::prove<TypeParam>(
          fri_config, transcript_config, scalars.get(), input_size, hash, compress, output_store_min_layer, fri_proof);
        ICICLE_CHECK(err);
        END_TIMER(FRIPROOF_sync, oss.str().c_str(), measure);

        // Release domain
        ICICLE_CHECK(ntt_release_domain<scalar_t>());

        // ===== Verifier side ======
        bool valid = false;
        err = fri_merkle_tree::verify<TypeParam>(fri_config, transcript_config, fri_proof, hash, compress, valid);
        ICICLE_CHECK(err);
        ASSERT_EQ(true, valid);

        // Serialize proof
        size_t proof_size = 0;
        ICICLE_CHECK(BinarySerializer<FriProof<TypeParam>>::serialized_size(fri_proof, proof_size));
        std::vector<std::byte> proof_bytes(proof_size);
        ICICLE_CHECK(
          BinarySerializer<FriProof<TypeParam>>::serialize(proof_bytes.data(), proof_bytes.size(), fri_proof));

        // Deserialize proof
        FriProof<TypeParam> deserialized_proof;
        ICICLE_CHECK(BinarySerializer<FriProof<TypeParam>>::deserialize(
          proof_bytes.data(), proof_bytes.size(), deserialized_proof));

        // Compare proofs
        // Compare number of FRI rounds
        ASSERT_EQ(fri_proof.get_nof_fri_rounds(), deserialized_proof.get_nof_fri_rounds());

        // Compare final polynomial size and contents
        ASSERT_EQ(fri_proof.get_final_poly_size(), deserialized_proof.get_final_poly_size());
        auto orig_final_poly_ptr = fri_proof.get_final_poly();
        auto deser_final_poly_ptr = deserialized_proof.get_final_poly();
        size_t orig_final_poly_size = fri_proof.get_final_poly_size();
        size_t deser_final_poly_size = deserialized_proof.get_final_poly_size();
        std::vector<TypeParam> orig_final_poly_vec(orig_final_poly_ptr, orig_final_poly_ptr + orig_final_poly_size);
        std::vector<TypeParam> deser_final_poly_vec(deser_final_poly_ptr, deser_final_poly_ptr + deser_final_poly_size);
        ASSERT_EQ(orig_final_poly_vec, deser_final_poly_vec);

        // Compare PoW nonce
        ASSERT_EQ(fri_proof.get_pow_nonce(), deserialized_proof.get_pow_nonce());

        // // Compare Merkle proofs for each query and round
        for (size_t query_idx = 0; query_idx < fri_proof.get_nof_fri_rounds(); query_idx++) {
          for (size_t round_idx = 0; round_idx < fri_proof.get_nof_fri_rounds(); round_idx++) {
            auto merkle_proof = fri_proof.get_query_proof_slot(query_idx, round_idx);
            auto deserialized_proof = fri_proof.get_query_proof_slot(query_idx, round_idx);
            ASSERT_EQ(merkle_proof.is_pruned(), deserialized_proof.is_pruned());

            // Compare paths
            auto [orig_path_ptr, orig_path_size] = merkle_proof.get_path();
            auto [deser_path_ptr, deser_path_size] = deserialized_proof.get_path();
            ASSERT_EQ(orig_path_size, deser_path_size);
            std::vector<std::byte> orig_path_vec(orig_path_ptr, orig_path_ptr + orig_path_size);
            std::vector<std::byte> deser_path_vec(deser_path_ptr, deser_path_ptr + deser_path_size);
            ASSERT_EQ(orig_path_vec, deser_path_vec);

            // Compare leaves
            auto [orig_leaf_ptr, orig_leaf_size, orig_leaf_idx] = merkle_proof.get_leaf();
            auto [deser_leaf_ptr, deser_leaf_size, deser_leaf_idx] = deserialized_proof.get_leaf();
            ASSERT_EQ(orig_leaf_size, deser_leaf_size);
            ASSERT_EQ(orig_leaf_idx, deser_leaf_idx);
            std::vector<std::byte> orig_leaf_vec(orig_leaf_ptr, orig_leaf_ptr + orig_leaf_size);
            std::vector<std::byte> deser_leaf_vec(deser_leaf_ptr, deser_leaf_ptr + deser_leaf_size);
            ASSERT_EQ(orig_leaf_vec, deser_leaf_vec);

            // Compare roots
            auto [orig_root_ptr, orig_root_size] = merkle_proof.get_root();
            auto [deser_root_ptr, deser_root_size] = deserialized_proof.get_root();
            ASSERT_EQ(orig_root_size, deser_root_size);
            std::vector<std::byte> orig_root_vec(orig_root_ptr, orig_root_ptr + orig_root_size);
            std::vector<std::byte> deser_root_vec(deser_root_ptr, deser_root_ptr + deser_root_size);
            ASSERT_EQ(orig_root_vec, deser_root_vec);
          }
        }
      };

      run(IcicleTestBase::reference_device(), false);
      run(IcicleTestBase::main_device(), false);
    }
  }
}

TYPED_TEST(FieldTest, FriShouldFailCases)
{
  const int log_input_size = 10;
  const int log_stopping_size = 4;
  const size_t pow_bits = 0;
  const size_t stopping_size = 1 << log_stopping_size;
  const size_t stopping_degree = stopping_size - 1;
  const uint64_t output_store_min_layer = 0;

  auto run = [stopping_degree, output_store_min_layer, pow_bits](
               const std::string& dev_type, const size_t nof_queries, const size_t folding_factor,
               const size_t log_domain_size, const size_t merkle_tree_arity, const size_t input_size) {
    // Generate input polynomial evaluations
    auto scalars = std::make_unique<TypeParam[]>(input_size);
    TypeParam::rand_host_many(scalars.get(), input_size);

    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    // Initialize ntt domain
    NTTInitDomainConfig init_domain_config = default_ntt_init_domain_config();
    ICICLE_CHECK(ntt_init_domain(scalar_t::omega(log_domain_size), init_domain_config));

    // ===== Prover side ======
    // Define hashers for merkle tree
    Hash hash = Keccak256::create(sizeof(TypeParam));                          // hash element -> 32B
    Hash compress = Keccak256::create(merkle_tree_arity * hash.output_size()); // hash every 64B to 32B

    const char* domain_separator_label = "domain_separator_label";
    const char* round_challenge_label = "round_challenge_label";
    const char* commit_phase_label = "commit_phase_label";
    const char* nonce_label = "nonce_label";
    std::vector<std::byte>&& public_state = {};
    TypeParam seed_rng = TypeParam::one();

    FriTranscriptConfig<TypeParam> transcript_config(
      hash, domain_separator_label, round_challenge_label, commit_phase_label, nonce_label, std::move(public_state),
      seed_rng);

    FriConfig fri_config;
    fri_config.nof_queries = nof_queries;
    fri_config.pow_bits = pow_bits;
    fri_config.folding_factor = folding_factor;
    fri_config.stopping_degree = stopping_degree;
    FriProof<TypeParam> fri_proof;

    eIcicleError error = prove_fri_merkle_tree<TypeParam>(
      fri_config, transcript_config, scalars.get(), input_size, hash, compress, output_store_min_layer, fri_proof);

    // Release domain
    ICICLE_CHECK(ntt_release_domain<scalar_t>());

    if (error == eIcicleError::SUCCESS) {
      // ===== Verifier side ======
      bool valid = false;
      error = verify_fri_merkle_tree<TypeParam>(fri_config, transcript_config, fri_proof, hash, compress, valid);
      ASSERT_EQ(true, valid);
    }
    ASSERT_EQ(error, eIcicleError::INVALID_ARGUMENT);
  };

  // Reference Device
  // Test invalid nof_queries
  run(
    IcicleTestBase::reference_device(), 0 /*nof_queries*/, 2 /*folding_factor*/, log_input_size /*log_domain_size*/,
    2 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  run(
    IcicleTestBase::main_device(), (1 << log_input_size) / 2 + 1 /*nof_queries*/, 2 /*folding_factor*/,
    log_input_size /*log_domain_size*/, 2 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  // Test invalid folding_factor  (currently not supported)
  run(
    IcicleTestBase::reference_device(), 10 /*nof_queries*/, 16 /*folding_factor*/, log_input_size /*log_domain_size*/,
    2 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  // Test too small domain size
  run(
    IcicleTestBase::reference_device(), 10 /*nof_queries*/, 2 /*folding_factor*/,
    log_input_size - 1 /*log_domain_size*/, 2 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  // Test invalid merkle tree arity
  run(
    IcicleTestBase::reference_device(), 10 /*nof_queries*/, 2 /*folding_factor*/, log_input_size /*log_domain_size*/,
    4 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  // Test invallid input size
  run(
    IcicleTestBase::reference_device(), 10 /*nof_queries*/, 2 /*folding_factor*/, log_input_size /*log_domain_size*/,
    2 /*merkle_tree_arity*/, (1 << log_input_size) - 1 /*input_size*/);

  // Main Device
  // Test invalid nof_queries
  run(
    IcicleTestBase::main_device(), 0 /*nof_queries*/, 2 /*folding_factor*/, log_input_size /*log_domain_size*/,
    2 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  run(
    IcicleTestBase::main_device(), (1 << log_input_size) / 2 + 1 /*nof_queries*/, 2 /*folding_factor*/,
    log_input_size /*log_domain_size*/, 2 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  // Test invalid folding_factor  (currently not supported)
  run(
    IcicleTestBase::main_device(), 10 /*nof_queries*/, 16 /*folding_factor*/, log_input_size /*log_domain_size*/,
    2 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  // Test too small domain size
  run(
    IcicleTestBase::main_device(), 10 /*nof_queries*/, 2 /*folding_factor*/,
    log_input_size - 1
    /*log_domain_size*/,
    2 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  // Test invalid merkle tree arity
  run(
    IcicleTestBase::main_device(), 10 /*nof_queries*/, 2 /*folding_factor*/,
    log_input_size
    /*log_domain_size*/,
    4 /*merkle_tree_arity*/, 1 << log_input_size /*input_size*/);
  // Test invallid input size
  run(
    IcicleTestBase::main_device(), 10 /*nof_queries*/, 2 /*folding_factor*/,
    log_input_size
    /*log_domain_size*/,
    4 /*merkle_tree_arity*/, (1 << log_input_size) - 1 /*input_size*/);
}

#endif // FRI

TEST_F(FieldTestBase, FieldStorageReduceSanityTest)
{
  /*
  SR - storage reduce
  check that:
  1. SR(x1) + SR(x1) = SR(x1+x2)
  2. SR(INV(SR(x))*x) = 1
  */
  START_TIMER(StorageSanity)
  for (int i = 0; i < 1000; i++) {
    storage<18> a =                                          // 18 because we support up to 576 bits
      scalar_t::template rand_storage<18>(17);               // 17 so we don't have carry after addition
    storage<18> b = scalar_t::template rand_storage<18>(17); // 17 so we don't have carry after addition
    storage<18> sum = {};
    const storage<18 - (scalar_t::TLC > 1 ? scalar_t::TLC : 2)> c =
      scalar_t::template rand_storage<18 - (scalar_t::TLC > 1 ? scalar_t::TLC : 2)>(); // -TLC so we don't overflow in
                                                                                       // multiplication
    storage<18> product = {};
    host_math::template add_sub_limbs<18, false, false, true>(a, b, sum);
    auto c_red = scalar_t::from(c);
    auto c_inv = scalar_t::inverse(c_red);
    storage<(scalar_t::TLC > 1 ? scalar_t::TLC : 2)> c_inv_s = {c_inv.limbs_storage.limbs[0]};
    if (scalar_t::TLC > 1) {
      for (int i = 1; i < scalar_t::TLC; i++) {
        c_inv_s.limbs[i] = c_inv.limbs_storage.limbs[i];
      }
    }
    host_math::multiply_raw(c, c_inv_s, product);
    ASSERT_EQ(scalar_t::from(a) + scalar_t::from(b), scalar_t::from(sum));
    ASSERT_EQ(scalar_t::from(product), scalar_t::one());
    std::byte* a_bytes = reinterpret_cast<std::byte*>(a.limbs);
    std::byte* b_bytes = reinterpret_cast<std::byte*>(b.limbs);
    std::byte* sum_bytes = reinterpret_cast<std::byte*>(sum.limbs);
    std::byte* product_bytes = reinterpret_cast<std::byte*>(product.limbs);
    ASSERT_EQ(scalar_t::from(a), scalar_t::from(a_bytes, 18 * 4));
    ASSERT_EQ(scalar_t::from(a_bytes, 18 * 4) + scalar_t::from(b_bytes, 18 * 4), scalar_t::from(sum_bytes, 18 * 4));
    ASSERT_EQ(scalar_t::from(product_bytes, 18 * 4), scalar_t::one());
  }
  END_TIMER(StorageSanity, "storage sanity", true);
}
