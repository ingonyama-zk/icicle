#include <iostream>
#include <cstring>
#include <chrono>
#include <string>
#include "hash/blake2/blake2s.h"
#include <chrono>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <algorithm>

using namespace icicle;

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

void print_hash(uint8_t* hash, unsigned int len)
{
  for (unsigned int i = 0; i < len; i++) {
    printf("%02x", hash[i]);
  }
  printf("\n");
}

// Function to trim leading and trailing whitespace from a string
std::string trim(const std::string& str)
{
  size_t first = str.find_first_not_of(' ');
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

std::unordered_map<std::string, std::string> load_strings_and_hashes_from_csv(const char* filename)
{
  std::unordered_map<std::string, std::string> string_hash_map;
  std::ifstream file(filename);
  if (!file.is_open()) {
    perror("Failed to open CSV file");
    exit(EXIT_FAILURE);
  }

  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string input_string, hash_value;
    if (std::getline(ss, input_string, ',') && std::getline(ss, hash_value, ',')) {
      // Trim any whitespace around the strings
      input_string = trim(input_string);
      hash_value = trim(hash_value);
      string_hash_map[input_string] = hash_value;
    }
  }

  file.close();
  return string_hash_map;
}

int main(int argc, char** argv)
{
  using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

  if (argc < 2) {
    fprintf(stderr, "Usage: %s <strings and hashes CSV file>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char* csv_filename = argv[1];
  auto string_hash_map = load_strings_and_hashes_from_csv(csv_filename);

  if (string_hash_map.size() != 10) {
    fprintf(stderr, "CSV file must contain exactly 10 strings and hashes.\n");
    return EXIT_FAILURE;
  }

  // Prepare the test strings and expected hashes from the map
  std::vector<std::string> test_strings;
  std::vector<std::string> expected_hashes;
  for (const auto& pair : string_hash_map) {
    test_strings.push_back(pair.first);
    expected_hashes.push_back(pair.second);
  }

  // Test parameters
  unsigned int n_batch = test_strings.size(); // Number of different inputs to hash in parallel
  size_t inlen = test_strings[0].size();      // Max length of the test strings

  // Calculate total input length and allocate memory for the batched input
  size_t total_len = 0;
  for (const auto& str : test_strings) {
    total_len += str.size();
  }
  uint8_t* batched_input = (uint8_t*)malloc(total_len);
  unsigned int* in_lengths = (unsigned int*)malloc(n_batch * sizeof(unsigned int));

  // Copy test strings to batched input
  uint8_t* current_position = batched_input;
  for (unsigned int i = 0; i < n_batch; ++i) {
    memcpy(current_position, test_strings[i].c_str(), test_strings[i].size());
    current_position += test_strings[i].size();
  }
  // init blake2s
  Blake2s blake2s = Blake2s(inlen / sizeof(limb_t));

  // Allocate memory for the output
  unsigned int outlen = blake2s.m_total_output_limbs * sizeof(limb_t);
  uint8_t* output = (uint8_t*)malloc(outlen * n_batch);
  if (!output) {
    perror("Failed to allocate memory for output");
    free(batched_input);
    free(in_lengths);
    return EXIT_FAILURE;
  }

  HashConfig config;
  // Perform the hashing
  START_TIMER(blake_timer)
  blake2s.run_multiple_hash((limb_t*)batched_input, (limb_t*)output, n_batch, config);

  END_TIMER(blake_timer, "Blake Timer")

  // Print and compare the results
  printf("BLAKE2S hash (batch size = %d):\n", n_batch);
  for (unsigned int i = 0; i < n_batch; i++) {
    // printf("String: %s\n", test_strings[i].c_str());
    // printf("Computed Hash %d: ", i + 1);
    // print_hash(output + i * outlen, outlen);
    // std::cout << "Expected Hash " << i + 1 << ": " << expected_hashes[i] << std::endl;

    std::string computed_hash;
    for (unsigned int j = 0; j < outlen; ++j) {
      char buffer[3];
      snprintf(buffer, sizeof(buffer), "%02x", output[i * outlen + j]);
      computed_hash += buffer;
    }

    if (computed_hash == expected_hashes[i]) {
      printf(" (Match)\n");
    } else {
      printf(" (Mismatch)\n");
    }
  }
  // Clean up
  free(output);
  free(batched_input);
  free(in_lengths);
  return 0;
}
