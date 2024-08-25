

#include <iostream>
#include <cstring>
#include <chrono>
#include <string>
#include "hash/blake2/blake2s.h"
#include <chrono>

using namespace blake2s_cpu;

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

// Ensure BLAKE2S_KEYBYTES, blake2s.total_output_limbs*sizeof(limb_t), and blake2s function are defined appropriately

int main(void)
{
  using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

  // The string to hash
  char* test_string_char = "01234567";
  size_t test_string_len = strlen(test_string_char);
  const BYTE* test_string = (BYTE*)test_string_char;
  HashConfig config; //= default_hash_config();
  // Test simple API
  Blake2s blake2s = Blake2s(test_string_len / sizeof(limb_t));
  START_TIMER(blake_ref)
  {
    uint8_t hash[blake2s.total_output_limbs * sizeof(limb_t)];
    // Pass the empty key with keylen = 0
    // virtual eIcicleError run_single_hash(const limb_t *input_limbs, limb_t *output_limbs, const HashConfig& config)

    blake2s.run_single_hash((limb_t*)test_string, (limb_t*)hash, config);

    // Expected hash value for "0123456789" with an empty key as a string
    const char* expected_hash_str = "0d74da2a1062445822cbc8ec7bf424714e09923b4c1eba0ca2170504f56c4331";

    // Convert computed hash to a string
    char computed_hash_str[blake2s.total_output_limbs * sizeof(limb_t) * 2 + 1]; // Two characters per byte + null
                                                                                 // terminator
    for (size_t i = 0; i < blake2s.total_output_limbs * sizeof(limb_t); i++) {
      sprintf(&computed_hash_str[i * 2], "%02x", hash[i]);
    }
    computed_hash_str[blake2s.total_output_limbs * sizeof(limb_t) * 2] = '\0'; // Null terminator

    // Print the computed and expected hash strings
    std::cout << "Computed hash: " << computed_hash_str << std::endl;
    std::cout << "Expected hash: " << expected_hash_str << std::endl;

    if (strcmp(computed_hash_str, expected_hash_str) != 0) {
      std::cerr << "Hash mismatch!" << std::endl;
      goto fail;
    }
  }
  END_TIMER(blake_ref, "Blake Ref Timer")

  puts("ok");
  return 0;
fail:
  puts("error");
  return -1;
}
