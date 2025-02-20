#include "test_mod_arithmetic_api.h"
#include "icicle/sumcheck/sumcheck.h"
#include "icicle/fri/fri.h"
#include "icicle/fri/fri_config.h"
#include "icicle/fri/fri_proof.h"
#include "icicle/fri/fri_transcript_config.h"

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

#ifdef NTT

TYPED_TEST(FieldApiTest, ntt)
{
  // Randomize configuration
  const bool inplace = rand_uint_32b(0, 1);
  const int logn = rand_uint_32b(0, 17);
  const uint64_t N = 1 << logn;
  const int log_ntt_domain_size = logn + 2; // Fix for Metal backend support
  const int log_batch_size = rand_uint_32b(0, 2);
  const int batch_size = 1 << log_batch_size;
  const int _ordering = rand_uint_32b(0, 3);
  const Ordering ordering = static_cast<Ordering>(_ordering);
  bool columns_batch;
  if (logn == 7 || logn < 4) {
    columns_batch = false; // currently not supported (icicle_v3/backend/cuda/src/ntt/ntt.cuh line 578)
  } else {
    columns_batch = rand_uint_32b(0, 1);
  }
  const NTTDir dir = static_cast<NTTDir>(rand_uint_32b(0, 1)); // 0: forward, 1: inverse
  const int log_coset_stride = rand_uint_32b(0, 2);
  scalar_t coset_gen;
  if (log_coset_stride) {
    coset_gen = scalar_t::omega(logn + log_coset_stride);
  } else {
    coset_gen = scalar_t::one();
  }

  ICICLE_LOG_DEBUG << "N = " << N;
  ICICLE_LOG_DEBUG << "batch_size = " << batch_size;
  ICICLE_LOG_DEBUG << "columns_batch = " << columns_batch;
  ICICLE_LOG_DEBUG << "inplace = " << inplace;
  ICICLE_LOG_DEBUG << "ordering = " << _ordering;
  ICICLE_LOG_DEBUG << "log_coset_stride = " << log_coset_stride;

  const int total_size = N * batch_size;
  auto scalars = std::make_unique<TypeParam[]>(total_size);
  TypeParam::rand_host_many(scalars.get(), total_size);

  auto out_main = std::make_unique<TypeParam[]>(total_size);
  auto out_ref = std::make_unique<TypeParam[]>(total_size);
  auto run = [&](const std::string& dev_type, TypeParam* out, const char* msg, bool measure, int iters) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    icicleStreamHandle stream = nullptr;
    ICICLE_CHECK(icicle_create_stream(&stream));
    auto init_domain_config = default_ntt_init_domain_config();
    init_domain_config.stream = stream;
    init_domain_config.is_async = false;
    ConfigExtension ext;
    ext.set(CudaBackendConfig::CUDA_NTT_FAST_TWIDDLES_MODE, true);
    init_domain_config.ext = &ext;
    auto config = default_ntt_config<scalar_t>();
    config.stream = stream;
    config.coset_gen = coset_gen;
    config.batch_size = batch_size;       // default: 1
    config.columns_batch = columns_batch; // default: false
    config.ordering = ordering;           // default: kNN
    config.are_inputs_on_device = true;
    config.are_outputs_on_device = true;
    config.is_async = false;
    ICICLE_CHECK(ntt_init_domain(scalar_t::omega(log_ntt_domain_size), init_domain_config));
    TypeParam *d_in, *d_out;
    ICICLE_CHECK(icicle_malloc_async((void**)&d_in, total_size * sizeof(TypeParam), config.stream));
    ICICLE_CHECK(icicle_malloc_async((void**)&d_out, total_size * sizeof(TypeParam), config.stream));
    ICICLE_CHECK(icicle_copy_to_device_async(d_in, scalars.get(), total_size * sizeof(TypeParam), config.stream));
    std::ostringstream oss;
    oss << dev_type << " " << msg;
    START_TIMER(NTT_sync)
    for (int i = 0; i < iters; ++i) {
      if (inplace) {
        ICICLE_CHECK(ntt(d_in, N, dir, config, d_in));
      } else {
        ICICLE_CHECK(ntt(d_in, N, dir, config, d_out));
      }
    }
    END_TIMER(NTT_sync, oss.str().c_str(), measure);

    if (inplace) {
      ICICLE_CHECK(icicle_copy_to_host_async(out, d_in, total_size * sizeof(TypeParam), config.stream));
    } else {
      ICICLE_CHECK(icicle_copy_to_host_async(out, d_out, total_size * sizeof(TypeParam), config.stream));
    }
    ICICLE_CHECK(icicle_free_async(d_in, config.stream));
    ICICLE_CHECK(icicle_free_async(d_out, config.stream));
    ICICLE_CHECK(icicle_stream_synchronize(config.stream));
    ICICLE_CHECK(icicle_destroy_stream(stream));
    ICICLE_CHECK(ntt_release_domain<scalar_t>());
  };
  run(IcicleTestBase::main_device(), out_main.get(), "ntt", false /*=measure*/, 10 /*=iters*/); // warmup
  run(IcicleTestBase::reference_device(), out_ref.get(), "ntt", VERBOSE /*=measure*/, 10 /*=iters*/);
  run(IcicleTestBase::main_device(), out_main.get(), "ntt", VERBOSE /*=measure*/, 10 /*=iters*/);
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
}
#endif // NTT

// define program
using MlePoly = Symbol<scalar_t>;

// define program
using MlePoly = Symbol<scalar_t>;
void lambda_multi_result(std::vector<MlePoly>& vars)
{
  const MlePoly& A = vars[0];
  const MlePoly& B = vars[1];
  const MlePoly& C = vars[2];
  const MlePoly& EQ = vars[3];
  vars[4] = EQ * (A * B - C) + scalar_t::from(9);
  vars[5] = A * B - C.inverse();
  vars[6] = vars[5];
}

TEST_F(FieldApiTestBase, CpuProgramExecutorMultiRes)
{
  scalar_t a = scalar_t::rand_host();
  scalar_t b = scalar_t::rand_host();
  scalar_t c = scalar_t::rand_host();
  scalar_t eq = scalar_t::rand_host();
  scalar_t res_0;
  scalar_t res_1;
  scalar_t res_2;

  Program<scalar_t> program(lambda_multi_result, 7);
  CpuProgramExecutor<scalar_t> prog_exe(program);

  // init program
  prog_exe.m_variable_ptrs[0] = &a;
  prog_exe.m_variable_ptrs[1] = &b;
  prog_exe.m_variable_ptrs[2] = &c;
  prog_exe.m_variable_ptrs[3] = &eq;
  prog_exe.m_variable_ptrs[4] = &res_0;
  prog_exe.m_variable_ptrs[5] = &res_1;
  prog_exe.m_variable_ptrs[6] = &res_2;

  // execute
  prog_exe.execute();

  // check correctness
  scalar_t expected_res_0 = eq * (a * b - c) + scalar_t::from(9);
  ASSERT_EQ(res_0, expected_res_0);

  scalar_t expected_res_1 = a * b - scalar_t::inverse(c);
  ASSERT_EQ(res_1, expected_res_1);
  ASSERT_EQ(res_2, res_1);
}

MlePoly returning_value_func(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  const MlePoly& EQ = inputs[3];
  return (EQ * (A * B - C));
}

TEST_F(FieldApiTestBase, CpuProgramExecutorReturningVal)
{
  // randomize input vectors
  const int total_size = 100000;
  auto in_a = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_a.get(), total_size);
  auto in_b = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_b.get(), total_size);
  auto in_c = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_c.get(), total_size);
  auto in_eq = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_eq.get(), total_size);

  //----- element wise operation ----------------------
  auto out_element_wise = std::make_unique<scalar_t[]>(total_size);
  START_TIMER(element_wise_op)
  for (int i = 0; i < 100000; ++i) {
    out_element_wise[i] = in_eq[i] * (in_a[i] * in_b[i] - in_c[i]);
  }
  END_TIMER(element_wise_op, "Straight forward function (Element wise) time: ", true);

  //----- explicit program ----------------------
  ReturningValueProgram<scalar_t> program_explicit(returning_value_func, 4);

  CpuProgramExecutor<scalar_t> prog_exe_explicit(program_explicit);
  auto out_explicit_program = std::make_unique<scalar_t[]>(total_size);

  // init program
  prog_exe_explicit.m_variable_ptrs[0] = in_a.get();
  prog_exe_explicit.m_variable_ptrs[1] = in_b.get();
  prog_exe_explicit.m_variable_ptrs[2] = in_c.get();
  prog_exe_explicit.m_variable_ptrs[3] = in_eq.get();
  prog_exe_explicit.m_variable_ptrs[4] = out_explicit_program.get();

  // run on all vectors
  START_TIMER(explicit_program)
  for (int i = 0; i < total_size; ++i) {
    prog_exe_explicit.execute();
    (prog_exe_explicit.m_variable_ptrs[0])++;
    (prog_exe_explicit.m_variable_ptrs[1])++;
    (prog_exe_explicit.m_variable_ptrs[2])++;
    (prog_exe_explicit.m_variable_ptrs[3])++;
    (prog_exe_explicit.m_variable_ptrs[4])++;
  }
  END_TIMER(explicit_program, "Explicit program executor time: ", true);

  // check correctness
  ASSERT_EQ(0, memcmp(out_element_wise.get(), out_explicit_program.get(), total_size * sizeof(scalar_t)));

  //----- predefined program ----------------------
  Program<scalar_t> predef_program(EQ_X_AB_MINUS_C);

  CpuProgramExecutor<scalar_t> prog_exe_predef(predef_program);
  auto out_predef_program = std::make_unique<scalar_t[]>(total_size);

  // init program
  prog_exe_predef.m_variable_ptrs[0] = in_a.get();
  prog_exe_predef.m_variable_ptrs[1] = in_b.get();
  prog_exe_predef.m_variable_ptrs[2] = in_c.get();
  prog_exe_predef.m_variable_ptrs[3] = in_eq.get();
  prog_exe_predef.m_variable_ptrs[4] = out_predef_program.get();

  // run on all vectors
  START_TIMER(predef_program)
  for (int i = 0; i < total_size; ++i) {
    prog_exe_predef.execute();
    (prog_exe_predef.m_variable_ptrs[0])++;
    (prog_exe_predef.m_variable_ptrs[1])++;
    (prog_exe_predef.m_variable_ptrs[2])++;
    (prog_exe_predef.m_variable_ptrs[3])++;
    (prog_exe_predef.m_variable_ptrs[4])++;
  }
  END_TIMER(predef_program, "Program predefined time: ", true);

  // check correctness
  ASSERT_EQ(0, memcmp(out_element_wise.get(), out_predef_program.get(), total_size * sizeof(scalar_t)));

  //----- Vecops operation ----------------------
  auto config = default_vec_ops_config();
  auto out_vec_ops = std::make_unique<scalar_t[]>(total_size);

  START_TIMER(vecop)
  vector_mul(in_a.get(), in_b.get(), total_size, config, out_vec_ops.get());         // A * B
  vector_sub(out_vec_ops.get(), in_c.get(), total_size, config, out_vec_ops.get());  // A * B - C
  vector_mul(out_vec_ops.get(), in_eq.get(), total_size, config, out_vec_ops.get()); // EQ * (A * B - C)
  END_TIMER(vecop, "Vec ops time: ", true);

  // check correctness
  ASSERT_EQ(0, memcmp(out_element_wise.get(), out_vec_ops.get(), total_size * sizeof(scalar_t)));
}

MlePoly ex_x_ab_minus_c_func(const std::vector<MlePoly>& inputs)
{
  const MlePoly& A = inputs[0];
  const MlePoly& B = inputs[1];
  const MlePoly& C = inputs[2];
  const MlePoly& EQ = inputs[3];
  return EQ * (A * B - C);
}

TEST_F(FieldApiTestBase, ProgramExecutorVecOp)
{
  // randomize input vectors
  const int total_size = 100000;
  const ReturningValueProgram<scalar_t> prog(ex_x_ab_minus_c_func, 4);
  auto in_a = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_a.get(), total_size);
  auto in_b = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_b.get(), total_size);
  auto in_c = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_c.get(), total_size);
  auto in_eq = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_eq.get(), total_size);

  auto run = [&](
               const std::string& dev_type, std::vector<scalar_t*>& data, const Program<scalar_t>& program,
               uint64_t size, const char* msg) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);
    auto config = default_vec_ops_config();

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(executeProgram)
    ICICLE_CHECK(execute_program(data, program, size, config));
    END_TIMER(executeProgram, oss.str().c_str(), true);
  };

  // initialize data vector for main device
  auto out_main = std::make_unique<scalar_t[]>(total_size);
  std::vector<scalar_t*> data_main = std::vector<scalar_t*>(5);
  data_main[0] = in_a.get();
  data_main[1] = in_b.get();
  data_main[2] = in_c.get();
  data_main[3] = in_eq.get();
  data_main[4] = out_main.get();

  // initialize data vector for reference device
  auto out_ref = std::make_unique<scalar_t[]>(total_size);
  std::vector<scalar_t*> data_ref = std::vector<scalar_t*>(5);
  data_ref[0] = in_a.get();
  data_ref[1] = in_b.get();
  data_ref[2] = in_c.get();
  data_ref[3] = in_eq.get();
  data_ref[4] = out_ref.get();

  // run on both devices and compare
  run(IcicleTestBase::main_device(), data_main, prog, total_size, "execute_program");
  run(IcicleTestBase::reference_device(), data_ref, prog, total_size, "execute_program");
  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
}

TEST_F(FieldApiTestBase, ProgramExecutorVecOpDataOnDevice)
{
  // randomize input vectors
  const int total_size = 100000;
  const int num_of_params = 5;
  const ReturningValueProgram<scalar_t> prog(ex_x_ab_minus_c_func, num_of_params - 1);
  auto in_a = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_a.get(), total_size);
  auto in_b = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_b.get(), total_size);
  auto in_c = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_c.get(), total_size);
  auto in_eq = std::make_unique<scalar_t[]>(total_size);
  scalar_t::rand_host_many(in_eq.get(), total_size);

  auto run = [&](
               const std::string& dev_type, std::vector<scalar_t*>& data, const Program<scalar_t>& program,
               uint64_t size, VecOpsConfig config, const char* msg) {
    Device dev = {dev_type, 0};
    icicle_set_device(dev);

    std::ostringstream oss;
    oss << dev_type << " " << msg;

    START_TIMER(executeProgram)
    ICICLE_CHECK(execute_program(data, program, size, config));
    END_TIMER(executeProgram, oss.str().c_str(), true);
  };

  // initialize data vector for main device
  auto out_main = std::make_unique<scalar_t[]>(total_size);
  std::vector<scalar_t*> data_main = std::vector<scalar_t*>(num_of_params);
  data_main[0] = in_a.get();
  data_main[1] = in_b.get();
  data_main[2] = in_c.get();
  data_main[3] = in_eq.get();
  data_main[4] = out_main.get();

  // initialize data vector for reference device
  auto out_ref = std::make_unique<scalar_t[]>(total_size);
  std::vector<scalar_t*> data_ref = std::vector<scalar_t*>(num_of_params);
  data_ref[0] = in_a.get();
  data_ref[1] = in_b.get();
  data_ref[2] = in_c.get();
  data_ref[3] = in_eq.get();
  data_ref[4] = out_ref.get();

  auto config = default_vec_ops_config();
  config.is_a_on_device = 1;

  // run on both devices and compare
  run(IcicleTestBase::reference_device(), data_ref, prog, total_size, config, "execute_program");

  icicle_set_device(IcicleTestBase::main_device());

  if (config.is_a_on_device) {
    for (int idx = 0; idx < num_of_params; ++idx) {
      scalar_t* tmp = nullptr;
      icicle_malloc((void**)&tmp, total_size * sizeof(scalar_t));
      icicle_copy_to_device(tmp, data_main[idx], total_size * sizeof(scalar_t));
      data_main[idx] = tmp;
    }
  }

  run(IcicleTestBase::main_device(), data_main, prog, total_size, config, "execute_program");

  if (config.is_a_on_device)
    icicle_copy_to_host(out_main.get(), data_main[num_of_params - 1], total_size * sizeof(scalar_t));

  ASSERT_EQ(0, memcmp(out_main.get(), out_ref.get(), total_size * sizeof(scalar_t)));
}

TEST_F(FieldApiTestBase, Sumcheck)
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
  oss << IcicleTestBase::main_device() << " " << "Sumcheck";

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
