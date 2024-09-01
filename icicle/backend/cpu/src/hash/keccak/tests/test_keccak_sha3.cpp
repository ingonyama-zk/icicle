#include <iostream>
#include <cstring>
#include <chrono>
#include <string>
#include "hash/keccak/keccak.h"

using namespace icicle;

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

int run_keccak_test(
  const Keccak_cpu& keccak_hash, const uint8_t* test_string, size_t test_string_len, const char* expected_hash_str)
{
  HashConfig config;
  uint8_t hash[keccak_hash.m_total_output_limbs * sizeof(limb_t)];

  keccak_hash.run_single_hash((limb_t*)test_string, (limb_t*)hash, config);

  char computed_hash_str[keccak_hash.m_total_output_limbs * sizeof(limb_t) * 2 + 1];
  for (size_t i = 0; i < keccak_hash.m_total_output_limbs * sizeof(limb_t); i++) {
    sprintf(&computed_hash_str[i * 2], "%02x", hash[i]);
  }
  computed_hash_str[keccak_hash.m_total_output_limbs * sizeof(limb_t) * 2] = '\0';

  std::cout << "Computed hash: " << computed_hash_str << std::endl;
  std::cout << "Expected hash: " << expected_hash_str << std::endl;

  if (strcmp(computed_hash_str, expected_hash_str) != 0) {
    std::cerr << "Hash mismatch!" << std::endl;
    return -1;
  }

  return 0;
}

int main(void)
{
  using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

  char* test_string_char = "01234567";
  size_t test_string_len = strlen(test_string_char);
  const uint8_t* test_string = (uint8_t*)test_string_char;

  const char* expected_hash_keccak256 = "d529b8ccadec912a5c302a7a9ef53e70c144eea6043dcea534fdbbb2d042fc31";
  const char* expected_hash_keccak512 = "a113f22ee81975d641b7ace6fc2fe4d865410dc5a56b3aa0a08ec57ad6400c1234ce27872e481a"
                                        "b3ba8ec8f5357e806d104b94122a555ce8bd20486ac8e778a1";
  const char* expected_hash_sha3_256 = "3f1347c106caaf7be58f3fc4f145bc6c2588fc1a5e85d0a2d27df55673d09d5d";
  const char* expected_hash_sha3_512 = "2ce3c81d4d11d2d97953f6c66db71e4bebf889d75610bdc132a3cd023c045049e83760b46913991"
                                       "91f999420cf7fa561188839723be2f98787240fe07f3dcb8b";

  bool success = true;

  printf("Testing Keccak256\n");
  START_TIMER(keccak_256)
  {
    Keccak_cpu keccak = Keccak256_cpu(test_string_len / sizeof(limb_t));
    if (run_keccak_test(keccak, test_string, test_string_len, expected_hash_keccak256) != 0) { success = false; }
  }
  END_TIMER(keccak_256, "Keccak-256 Timer")

  if (success) {
    printf("Testing Keccak512\n");
    START_TIMER(keccak_512)
    {
      Keccak_cpu keccak = Keccak512_cpu(test_string_len / sizeof(limb_t));
      if (run_keccak_test(keccak, test_string, test_string_len, expected_hash_keccak512) != 0) { success = false; }
    }
    END_TIMER(keccak_512, "Keccak-512 Timer")
  }

  if (success) {
    printf("Testing SHA3_256\n");
    START_TIMER(sha3_256)
    {
      Keccak_cpu keccak = Sha3_256_cpu(test_string_len / sizeof(limb_t));
      if (run_keccak_test(keccak, test_string, test_string_len, expected_hash_sha3_256) != 0) { success = false; }
    }
    END_TIMER(sha3_256, "SHA3-256 Timer")
  }

  if (success) {
    printf("Testing SHA3_512\n");
    START_TIMER(sha3_512)
    {
      Keccak_cpu keccak = Sha3_512_cpu(test_string_len / sizeof(limb_t));
      if (run_keccak_test(keccak, test_string, test_string_len, expected_hash_sha3_512) != 0) { success = false; }
    }
    END_TIMER(sha3_512, "SHA3-512 Timer")
  }

  if (success) {
    puts("ok");
    return 0;
  } else {
    puts("error!");
    return -1;
  }
}