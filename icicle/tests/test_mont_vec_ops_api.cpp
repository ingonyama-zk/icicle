
#include <gtest/gtest.h>
#include <iostream>

#include "icicle/runtime.h"
#include "icicle/memory_tracker.h"
// #include "dlfcn.h"

#include "icicle/api/bn254.h"
// #include "vec_ops/vec_ops.cuh"
#include "icicle/vec_ops.h"
// #include <vec_ops/../../include/utils/mont.cuh>

#include "icicle/fields/field_config.h"
#include "icicle/errors.h"
#include <nvml.h>

using namespace field_config;
// using namespace vec_ops;
using namespace bn254;
using namespace icicle;

typedef scalar_t T;

enum Op { MUL, ADD, SUB, LAST };

// p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001 # bn254

using FpNanoseconds = std::chrono::duration<float, std::chrono::nanoseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg, enable, iters)                                                                           \
  if (enable)                                                                                                          \
    printf(                                                                                                            \
      "%s: %.3f ns\n", msg, FpNanoseconds(std::chrono::high_resolution_clock::now() - timer##_start).count() / iters);

using namespace icicle;

int mont_test_for_dev(icicle::Device& dev);

int vector_op(
  T* vec_a,
  T* vec_b,
  T* vec_result,
  int n_elements,
  VecOpsConfig config,
  Op op)
{
  icicle::eIcicleError err;
  switch (op) {
  case MUL:
    err = vector_mul(vec_a, vec_b, n_elements, config, vec_result);
    break;
  case ADD:
    err = vector_add(vec_a, vec_b, n_elements, config, vec_result);
    break;
  case SUB:
    err = vector_sub(vec_a, vec_b, n_elements, config, vec_result);
    break;
  }
  EXPECT_EQ(err, icicle::eIcicleError::SUCCESS);
  return 0;
}

class MontVecOpsApiTest : public ::testing::Test
{
protected:
  static inline std::vector<std::string> s_registered_devices;
  // SetUpTestSuite/TearDownTestSuite are called once for the entire test suite
  static void SetUpTestSuite()
  {
#ifdef BACKEND_BUILD_DIR
    setenv("ICICLE_BACKEND_INSTALL_DIR", BACKEND_BUILD_DIR, 0 /*=replace*/);
#endif
    icicle_load_backend_from_env_or_default();
    s_registered_devices = get_registered_devices_list();
    ASSERT_GT(s_registered_devices.size(), 1);
  }
  static void TearDownTestSuite() {}

  // SetUp/TearDown are called before and after each test
  void SetUp() override {}
  void TearDown() override {}

};

TEST_F(MontVecOpsApiTest, UnregisteredDeviceError)
{
  icicle::Device dev = {"INVALID_DEVICE", 2};
  EXPECT_ANY_THROW(get_deviceAPI(dev));
}

TEST_F(MontVecOpsApiTest, CheckMulAddSubWithMontgomery)
{
  Device device_cuda = {"CUDA", 0};
  icicle_set_device(device_cuda);
  mont_test_for_dev(device_cuda);
  // The following code to be used when the test runs both for GPU and CPU. Not to remove for now.
  // for (const auto& device_type : s_registered_devices) {
  //   Device dev = {device_type, 0};
  //   std::cout << "Test DEVICE = " << dev.type << std::endl;
  //   ICICLE_CHECK(icicle_set_device(dev));
  //   mont_test_for_dev(dev);
  // }
}

int mont_test_for_dev(icicle::Device& dev)
{
  // const uint64_t vector_size = 1 << 15;    DEBUG
  // const unsigned not_in_place_repetitions = 1 << 10; // Repetitions are used only for the non in-place tests.    DEBUG
  const uint64_t vector_size = 1 << 15;
  const unsigned not_in_place_repetitions = 1 << 10; // Repetitions are used only for the non in-place tests.
  const unsigned in_place_repetitions = 1;           // Repetitions for in-place tests should be 1. Don't check it.

  icicle::eIcicleError err;
  std::cout << "Icicle-Examples: vector mul / add / sub operations." << std::endl;

  std::cout << "Vector size:              " << vector_size << std::endl;
  std::cout << "not_in_place_repetitions: " << not_in_place_repetitions << std::endl;
  std::cout << "in_place_repetitions:     " << in_place_repetitions << std::endl;

  // host data
  std::cout << "Allocate memory for the input vectors (both normal and Montgomery presentation)" << std::endl;
  T* host_in1_init = (T*)malloc(vector_size * sizeof(T));
  T* host_in2_init = (T*)malloc(vector_size * sizeof(T));
  std::cout << "Initializing vectors with normal presentation random data" << std::endl;
  T::rand_host_many(host_in1_init, vector_size);
  T::rand_host_many(host_in2_init, vector_size);
  std::cout << "Allocate memory for the output vectors" << std::endl;
  T* host_out = (T*)malloc(vector_size * sizeof(T)); // This memory will be used for the test output.
  T* host_out_ref_mul = (T*)malloc(
    vector_size *
    sizeof(T)); // This memory will be used as a reference result for mul (will be compared to host_out content).
  T* host_out_ref_add = (T*)malloc(
    vector_size *
    sizeof(T)); // This memory will be used as a reference result for add (will be compared to host_out content).
  T* host_out_ref_sub = (T*)malloc(
    vector_size *
    sizeof(T)); // This memory will be used as a reference result for sub (will be compared to host_out content).
  std::cout << "Initializing output vectors with random data" << std::endl;
  T::rand_host_many(host_out, vector_size);
  T::rand_host_many(host_out_ref_mul, vector_size);
  T::rand_host_many(host_out_ref_add, vector_size);
  T::rand_host_many(host_out_ref_sub, vector_size);
  T* device_in1 = nullptr;
  T* device_in2 = nullptr;;
  T* device_out = nullptr;;
  // Allocate memory on device.
  ICICLE_CHECK(icicle_malloc((void**)&device_in1, vector_size * sizeof(T)));
  ICICLE_CHECK(icicle_malloc((void**)&device_in2, vector_size * sizeof(T)));
  ICICLE_CHECK(icicle_malloc((void**)&device_out, vector_size * sizeof(T)));

  VecOpsConfig config = default_vec_ops_config();
  config.is_a_on_device = true;
  config.is_b_on_device = true;
  config.is_result_on_device = true;

  int nof_of_configs_for_test = 5;
  int nof_of_storage_configs = 3; // 2 inputs, 1 result.

  DeviceProperties device_props;
  icicle_get_device_properties(device_props);

  //****************************************
  // Test warn-up and reference output config. Reference output to be used to check if test passed or not.
  //****************************************
  if (!device_props.using_host_memory) {
    // copy from host to device
    ICICLE_CHECK(icicle_copy_to_device(device_in1, host_in1_init, vector_size * sizeof(T)));
    ICICLE_CHECK(icicle_copy_to_device(device_in2, host_in2_init, vector_size * sizeof(T)));
    std::cout << "Starting warm-up run" << std::endl;
    // Warm-up loop
    for (int i = 0; i < 100; i++) { // Nof loops set to 100 because warm-up takes too much time because inputs and outputs
                                    // are on located on Host.
      vector_op(device_in1, device_in2, device_out, vector_size, config, MUL);
    }
    // Generate ref results for all ops
    for (int op = MUL; op != LAST; op++) {
      vector_op(device_in1, device_in2, device_out, vector_size, config, (Op)op);
      switch (op) {
      case MUL:
        ICICLE_CHECK(icicle_copy_to_host(host_out_ref_mul, device_out, vector_size * sizeof(T)));
        break;
      case ADD:
        ICICLE_CHECK(icicle_copy_to_host(host_out_ref_add, device_out, vector_size * sizeof(T)));
        break;
      case SUB:
        ICICLE_CHECK(icicle_copy_to_host(host_out_ref_sub, device_out, vector_size * sizeof(T)));
        break;
      }
    }
  } else {      // if (!device_props.using_host_memory) {
    // Generate ref results for all ops
    for (int op = MUL; op != LAST; op++) {
      vector_op(host_in1_init, host_in2_init, host_out, vector_size, config, (Op)op);
      switch (op) {
      case MUL:
        memcpy(host_out_ref_mul, host_out, vector_size * sizeof(T));
        break;
      case ADD:
        memcpy(host_out_ref_add, host_out, vector_size * sizeof(T));
        break;
      case SUB:
        memcpy(host_out_ref_sub, host_out, vector_size * sizeof(T));
        break;
      }
    }
  }
  //****************************************
  // End of test warn-up and reference output config.
  //****************************************

  //*******************************************************
  // Benchmark test:
  // Loop for (mul, add, sub):
  //   Loop (is_a_on_device, is_b_on_device, is_result_on_device, is_input_in_montgomery_form):
  //*******************************************************
  T* host_in1 =
    (T*)malloc(vector_size * sizeof(T)); // This buffer is used to load the data from host_in1_init for the benchmark.
  T* host_in2 =
    (T*)malloc(vector_size * sizeof(T)); // This buffer is used to load the data from host_in2_init for the benchmark.
  // Test when the result is not in-place
  std::cout << "**********************************************" << std::endl;
  std::cout << "*** Start " << dev.type << " not in-place benchmark loop ***" << std::endl;
  std::cout << "**********************************************" << std::endl;
  for (int op = MUL; op != LAST; op++) {
    for (int config_idx = 0; config_idx < 32; config_idx++) {
      switch (op) {
      case MUL:
        std::cout << "Start benchmark loop for op MUL config_idx " << config_idx << " not in-place" << std::endl;
        break;
      case ADD:
        std::cout << "Start benchmark loop for op ADD config_idx " << config_idx << " not in-place" << std::endl;
        break;
      case SUB:
        std::cout << "Start benchmark loop for op SUB config_idx " << config_idx << " not in-place" << std::endl;
        break;
      }
      // Destroy the result of the prev loop.
      T::rand_host_many(host_out, vector_size); // Randomize host_out in order to randomize device_out.
      ICICLE_CHECK(icicle_copy_to_device(device_out, host_out, vector_size * sizeof(T)));
      T::rand_host_many(host_out, vector_size); // Make host_out != device_out.
      // Initialize inputs with the known data
      for (int i = 0; i < vector_size; i++) {
        host_in1[i] = host_in1_init[i];
        host_in2[i] = host_in2_init[i];
      }
      config.is_a_on_device = (config_idx >> 4) & 0x1;
      config.is_b_on_device = (config_idx >> 3) & 0x1;
      config.is_result_on_device = (config_idx >> 2) & 0x1;
      config.is_input_in_montgomery_form = (config_idx >> 1) & 0x1;
      config.is_result_in_montgomery_form = (config_idx >> 0) & 0x1;

      // Copy from host to device (copy again in order to be used later in the loop and device_inX was already
      // overwritten by warmup.
      if (config.is_a_on_device) {
        if (config.is_input_in_montgomery_form) {
          ICICLE_CHECK(icicle_copy_to_device(device_in1, host_in1, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_in1, vector_size, true, config, device_in1));
          // Destroy host_in1 value with values of host_in2.
          for (int i = 0; i < vector_size; i++) {
            host_in1[i] = host_in2_init[i];
          }
        } else { // Normal presentation.
          ICICLE_CHECK(icicle_copy_to_device(device_in1, host_in1, vector_size * sizeof(T)));
        }
      } else {
        if (config.is_input_in_montgomery_form) { // Copy to device, cnvert to montgomery and copy back to host.
          ICICLE_CHECK(icicle_copy_to_device(device_in1, host_in1, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_in1, vector_size, true, config, device_in1));
          ICICLE_CHECK(icicle_copy_to_host(host_in1, device_in1, vector_size * sizeof(T)));
          // Destroy device_in1 value with values of host_in2.
          ICICLE_CHECK(icicle_copy_to_device(device_in1, host_in2, vector_size * sizeof(T)));
        }
      }

      if (config.is_b_on_device) {
        if (config.is_input_in_montgomery_form) {
          ICICLE_CHECK(icicle_copy_to_device(device_in2, host_in2, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_in2, vector_size, true, config, device_in2));
          for (int i = 0; i < vector_size; i++) {
            host_in2[i] = host_in1_init[i];
          }
        } else {
          // Normal presentation.
          ICICLE_CHECK(icicle_copy_to_device(device_in2, host_in2, vector_size * sizeof(T)));
        }
      } else {
        if (config.is_input_in_montgomery_form) { // Copy to device, cnvert to montgomery and copy back to host.
          ICICLE_CHECK(icicle_copy_to_device(device_in2, host_in2, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_in2, vector_size, true, config, device_in2));
          ICICLE_CHECK(icicle_copy_to_host(host_in2, device_in2, vector_size * sizeof(T)));
          // Destroy device_in2 valuewith values of host_in1.
          ICICLE_CHECK(icicle_copy_to_device(device_in2, host_in1, vector_size * sizeof(T)));
        }
      }

      if (device_props.using_host_memory && (config.is_a_on_device || config.is_b_on_device || config.is_result_on_device)) {
        std::cout << "    This config isn't suitable to run on CPU. Skip." << std::endl;
        continue;   // Proceed with the next config.
      }
      auto start_time = std::chrono::high_resolution_clock::now();
      switch (config_idx >> (nof_of_configs_for_test -
                             nof_of_storage_configs)) { // {is_a_on_device, is_b_on_device, is_result_on_device}
      case 0b000:
        for (int i = 0; i < not_in_place_repetitions; i++) {
          vector_op(host_in1, host_in2, host_out, vector_size, config, (Op)op);
        }
        break;
      case 0b001:
        for (int i = 0; i < not_in_place_repetitions; i++) {
          vector_op(host_in1, host_in2, device_out, vector_size, config, (Op)op);
        }
        break;
      case 0b010:
        for (int i = 0; i < not_in_place_repetitions; i++) {
          vector_op(host_in1, device_in2, host_out, vector_size, config, (Op)op);
        }
        break;
      case 0b011:
        for (int i = 0; i < not_in_place_repetitions; i++) {
          vector_op(host_in1, device_in2, device_out, vector_size, config, (Op)op);
        }
        break;
      case 0b100:
        for (int i = 0; i < not_in_place_repetitions; i++) {
          vector_op(device_in1, host_in2, host_out, vector_size, config, (Op)op);
        }
        break;
      case 0b101:
        for (int i = 0; i < not_in_place_repetitions; i++) {
          vector_op(device_in1, host_in2, device_out, vector_size, config, (Op)op);
        }
        break;
      case 0b110:
        for (int i = 0; i < not_in_place_repetitions; i++) {
          vector_op(device_in1, device_in2, host_out, vector_size, config, (Op)op);
        }
        break;
      case 0b111:
        for (int i = 0; i < not_in_place_repetitions; i++) {
          vector_op(device_in1, device_in2, device_out, vector_size, config, (Op)op);
        }
        break;
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      switch (op) {
      case MUL:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation MUL for config_idx "
                  << config_idx << " and result not in-place" << std::endl;
        break;
      case ADD:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation ADD for config_idx "
                  << config_idx << " and result not in-place" << std::endl;
        break;
      case SUB:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation SUB for config_idx "
                  << config_idx << " and result not in-place" << std::endl;
        break;
      }

      if (config.is_result_on_device) { // Copy the data to host_out in order to compare it vs. host_out_ref_XXX value.
        if (config.is_result_in_montgomery_form) { // Convert to normal from montgomery if needed.
          ICICLE_CHECK(convert_montgomery(device_out, vector_size, false, config, device_out));
        }
        ICICLE_CHECK(icicle_copy_to_host(host_out, device_out, vector_size * sizeof(T)));
      } else {                                     // Data is not on device but it is in host_out.
        if (config.is_result_in_montgomery_form) { // host_out should be written to device, converted to mmontgomery and
                                                   // written back to host. Then compared vs. host_out_ref_XXX.
          ICICLE_CHECK(icicle_copy_to_device(device_out, host_out, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_out, vector_size, false, config, device_out));
          ICICLE_CHECK(icicle_copy_to_host(host_out, device_out, vector_size * sizeof(T)));
        } else { // host_out could be compared vs. host_out_ref_XXX as is.
        }
      }
      //****************************************
      // End of benchmark test.
      //****************************************

      //***********************************************
      // Test result check (not in-place)
      // Check is performed by executing the operation in a normal presentation
      //   (located in in host_out_ref_XXX) and comparing it with the
      //   benchmark test result.
      //***********************************************
      switch (op) {
      case MUL:
        for (int i = 0; i < vector_size; i++) {
          EXPECT_EQ(host_out_ref_mul[i], host_out[i]);
        }
        break;
      case ADD:
        for (int i = 0; i < vector_size; i++) {
          EXPECT_EQ(host_out_ref_add[i], host_out[i]);
        }
        break;
      case SUB:
        for (int i = 0; i < vector_size; i++) {
          EXPECT_EQ(host_out_ref_sub[i], host_out[i]);
        }
        break;
      }
    }     // for (int config_idx = 0; config_idx < 32; config_idx++) {
  }     // for (int op = MUL; op != LAST; op++) {

  // Test when the result is in-place
  std::cout << "******************************************" << std::endl;
  std::cout << "*** Start " << dev.type << " in-place benchmark loop ***" << std::endl;
  std::cout << "******************************************" << std::endl;
  for (int op = MUL; op != LAST; op++) {
    for (int config_idx = 0; config_idx < 32; config_idx++) {
      switch (op) {
      case MUL:
        std::cout << "Start benchmark loop for op MUL config_idx " << config_idx << " in-place" << std::endl;
        break;
      case ADD:
        std::cout << "Start benchmark loop for op ADD config_idx " << config_idx << " in-place" << std::endl;
        break;
      case SUB:
        std::cout << "Start benchmark loop for op SUB config_idx " << config_idx << " in-place" << std::endl;
        break;
      }
      // Destroy the result of the prev loop.
      T::rand_host_many(host_out, vector_size); // Randomize host_out in order to randomize device_out.
      ICICLE_CHECK(icicle_copy_to_device(device_out, host_out, vector_size * sizeof(T)));
      T::rand_host_many(host_out, vector_size); // Make hist_out != device_out.
      // Initialize inputs with the known data. For in-place tests host_in1 is going to be used as a result. So, it
      // should be initialized later in the repetitions loop.
      for (int i = 0; i < vector_size; i++) {
        host_in1[i] = host_in1_init[i];
        host_in2[i] = host_in2_init[i];
      }
      config.is_a_on_device = (config_idx >> 4) & 0x1;
      config.is_b_on_device = (config_idx >> 3) & 0x1;
      config.is_result_on_device = (config_idx >> 2) & 0x1;
      config.is_input_in_montgomery_form = (config_idx >> 1) & 0x1;
      config.is_result_in_montgomery_form = (config_idx >> 1) & 0x1;
      if (config.is_a_on_device ^ config.is_result_on_device == 1) { continue; } // Illegal case for this loop.
      if (config.is_input_in_montgomery_form ^ config.is_result_in_montgomery_form == 1) {
        continue;
      } // Illegal case for this loop.

      // Copy from host to device (copy again in order to be used later in the loop and device_inX was already
      // overwritten by warmup.
      if (config.is_a_on_device) {
        if (config.is_input_in_montgomery_form) {
          ICICLE_CHECK(icicle_copy_to_device(device_in1, host_in1, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_in1, vector_size, true, config, device_in1));
          // Destroy host_in1 value with values of host_in2.
          for (int i = 0; i < vector_size; i++) {
            host_in1[i] = host_in2_init[i];
          }
        } else { // Normal presentation.
          ICICLE_CHECK(icicle_copy_to_device(device_in1, host_in1, vector_size * sizeof(T)));
        }
      } else {
        if (config.is_input_in_montgomery_form) { // Copy to device, cnvert to montgomery and copy back to host.
          ICICLE_CHECK(icicle_copy_to_device(device_in1, host_in1, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_in1, vector_size, true, config, device_in1));
          ICICLE_CHECK(icicle_copy_to_host(host_in1, device_in1, vector_size * sizeof(T)));
          // Destroy device_in1 value with values of host_in2.
          ICICLE_CHECK(icicle_copy_to_device(device_in1, host_in2, vector_size * sizeof(T)));
        }
      }

      if (config.is_b_on_device) {
        if (config.is_input_in_montgomery_form) {
          ICICLE_CHECK(icicle_copy_to_device(device_in2, host_in2, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_in2, vector_size, true, config, device_in2));
          // Destroy host_in2 value with values of host_in1.
          for (int i = 0; i < vector_size; i++) {
            host_in2[i] = host_in1_init[i];
          }
        } else {
          // Normal presentation.
          ICICLE_CHECK(icicle_copy_to_device(device_in2, host_in2, vector_size * sizeof(T)));
        }
      } else {
        if (config.is_input_in_montgomery_form) { // Copy to device, cnvert to montgomery and copy back to host.
          ICICLE_CHECK(icicle_copy_to_device(device_in2, host_in2, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_in2, vector_size, true, config, device_in2));
          ICICLE_CHECK(icicle_copy_to_host(host_in2, device_in2, vector_size * sizeof(T)));
          // Destroy device_in2 value with values of host_in1.
          ICICLE_CHECK(icicle_copy_to_device(device_in2, host_in1, vector_size * sizeof(T)));
        }
      }

      if (device_props.using_host_memory && (config.is_a_on_device || config.is_b_on_device || config.is_result_on_device)) {
        std::cout << "    This config isn't suitable to run on CPU. Skip." << std::endl;
        continue;   // Proceed with the next config.
      }
      auto start_time = std::chrono::high_resolution_clock::now();
      // Benchmark loop
      for (int i = 0; i < in_place_repetitions; i++) {
        switch (config_idx >> (nof_of_configs_for_test -
                               nof_of_storage_configs)) { // {is_a_on_device, is_b_on_device, is_result_on_device}
        case 0b000:
          vector_op(host_in1, host_in2, host_in1, vector_size, config, (Op)op);      
          break;
        case 0b001:
          break;
        case 0b010:
          vector_op(host_in1, device_in2, host_in1, vector_size, config, (Op)op);
          break;
        case 0b011:
          break;
        case 0b100:
          break;
        case 0b101:
          vector_op(device_in1, host_in2, device_in1, vector_size, config, (Op)op);
          break;
        case 0b110:
          break;
        case 0b111:
          vector_op(device_in1, device_in2, device_in1, vector_size, config, (Op)op);
          break;
        }
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      switch (op) {
      case MUL:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation MUL for config_idx "
                  << config_idx << " and result in-place" << std::endl;
        break;
      case ADD:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation ADD for config_idx "
                  << config_idx << " and result in-place" << std::endl;
        break;
      case SUB:
        std::cout << "Elapsed time: " << duration.count() << " microseconds, operation SUB for config_idx "
                  << config_idx << " and result in-place" << std::endl;
        break;
      }

      if (config.is_result_on_device) { // Copy the data to host_out in order to compare it vs. host_out_ref_XXX value.
        if (config.is_result_in_montgomery_form) { // Convert to normal from montgomery if needed.
          ICICLE_CHECK(convert_montgomery(device_in1, vector_size, false, config, device_in1));
        }
        ICICLE_CHECK(icicle_copy_to_host(host_out, device_in1, vector_size * sizeof(T)));
      } else { // Data is not on device but it is in host_in1. It should be moved to host_out for test pass/fail check.
        if (config.is_result_in_montgomery_form) { // host_out should be written to device, converted to mmontgomery and
                                                   // written back to host. Then compared vs. host_out_ref_XXX.
          ICICLE_CHECK(icicle_copy_to_device(device_out, host_in1, vector_size * sizeof(T)));
          ICICLE_CHECK(convert_montgomery(device_out, vector_size, false, config, device_out));
          ICICLE_CHECK(icicle_copy_to_host(host_out, device_out, vector_size * sizeof(T)));
        } else { // host_out could be compared vs. host_out_ref_XXX as is.
          for (int i = 0; i < vector_size;
               i++) { // Copy from host_in1 (result) to host_out to compare later vs. host_out_ref_XXX.
            host_out[i] = host_in1[i];
          }
        }
      }
      //****************************************
      // End of benchmark test.
      //****************************************

      //***********************************************
      // Test result check (in-place)
      // Check is performed by executing the operation in a normal presentation
      //   (located in in host_out_ref_XXX) and comparing it with the
      //   benchmark test result.
      //***********************************************
      switch (op) {
      case MUL:
        for (int i = 0; i < vector_size; i++) {
          EXPECT_EQ(host_out_ref_mul[i], host_out[i]);
        }
        break;
      case ADD:
        for (int i = 0; i < vector_size; i++) {
          EXPECT_EQ(host_out_ref_add[i], host_out[i]);
        }
        break;
      case SUB:
        for (int i = 0; i < vector_size; i++) {
          EXPECT_EQ(host_out_ref_sub[i], host_out[i]);
        }
        break;
      }
    }
  }
  return 0;
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
