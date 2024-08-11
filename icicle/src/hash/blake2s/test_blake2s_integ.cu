#include <chrono>
#include "gpu-utils/device_context.cuh"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "extern.cu"

using namespace blake2s;

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg)                                                                                          \
  printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

extern "C" {
void mcm_cuda_blake2s_hash_batch(BYTE* key, WORD keylen, BYTE* in, WORD inlen, BYTE* out, WORD n_outbit, WORD n_batch);
}

void print_hash(BYTE* hash, WORD len)
{
  printf("Hash Len: %d \n", len);
  printf("BLAKE2S hash:\n");
  for (WORD i = 0; i < len; i++) {
    printf("%02x", hash[i]);
  }
  printf("\n");
}

BYTE* read_file(const char* filename, size_t* filesize)
{
  FILE* file = fopen(filename, "rb");
  if (!file) {
    perror("Failed to open file");
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  *filesize = ftell(file);
  fseek(file, 0, SEEK_SET);

  BYTE* buffer = (BYTE*)malloc(*filesize);
  if (!buffer) {
    perror("Failed to allocate memory");
    fclose(file);
    exit(EXIT_FAILURE);
  }

  size_t bytesRead = fread(buffer, 1, *filesize, file);
  if (bytesRead != *filesize) {
    perror("Failed to read file");
    free(buffer);
    fclose(file);
    exit(EXIT_FAILURE);
  }

  fclose(file);
  return buffer;
}

int main(int argc, char** argv)
{
  using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
  using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

  BYTE* input;
  size_t inlen;
  const char* input_filename;
  const char* default_input = "aaaaaaaaaaa";

  if (argc < 2) {
    // Use default input if no file is provided
    input = (BYTE*)default_input;
    inlen = strlen(default_input);
  } else {
    input_filename = argv[1];
    input = read_file(input_filename, &inlen);
  }

  // Test parameters
  BYTE key[32] = ""; // Example key
  WORD keylen = strlen((char*)key);
  WORD n_outbit = 256; // Output length in bits
  WORD n_batch = 1;    // Number of hashes to compute in parallel
  // Allocate memory for the output
  WORD outlen = n_outbit / 8;
  BYTE* output = (BYTE*)malloc(outlen * n_batch);
  if (!output) {
    perror("Failed to allocate memory for output");
    if (argc >= 2) free(input); // Free file buffer if it was allocated
    return EXIT_FAILURE;
  }

  printf("Key len: %d \n", keylen);

  // Perform the hashing
  START_TIMER(blake_timer)
  HashConfig config = default_hash_config();

  blake2s_cuda(input, output, n_batch, inlen, outlen, config);
  END_TIMER(blake_timer, "Blake Timer")

  // Print the result
  print_hash(output, outlen);

  // Clean up
  free(output);
  if (argc >= 2) free(input); // Free file buffer if it was allocated
  return 0;
}
