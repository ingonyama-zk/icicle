#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "blake2s.cuh"
#include <chrono>

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg) \
  printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

extern "C" {
void mcm_cuda_blake2s_hash_batch(BYTE *key, WORD keylen, BYTE *in, WORD inlen, BYTE *out, WORD n_outbit, WORD n_batch);
}

void print_hash(BYTE *hash, WORD len) {
    for (WORD i = 0; i < len; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

BYTE *read_file(const char *filename, size_t *filesize) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    *filesize = ftell(file);
    fseek(file, 0, SEEK_SET);

    BYTE *buffer = (BYTE *)malloc(*filesize);
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

int main(int argc, char **argv) {
    using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
    using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input file 1> <input file 2>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_filename1 = argv[1];
    const char *input_filename2 = argv[2];

    // Read the first file
    size_t inlen1;
    BYTE *input1 = read_file(input_filename1, &inlen1);

    // Read the second file
    size_t inlen2;
    BYTE *input2 = read_file(input_filename2, &inlen2);

    // Test parameters
    BYTE key[32] = "";  // Example key
    WORD keylen = strlen((char *)key);
    WORD n_outbit = 256;  // Output length in bits
    WORD n_batch = 2;  // Number of different inputs to hash in parallel

    // Allocate memory for the batched input
    size_t inlen = (inlen1 > inlen2) ? inlen1 : inlen2;  // Ensure the buffer size can hold the larger input
    BYTE *batched_input = (BYTE *)malloc(inlen * n_batch);
    memset(batched_input, 0, inlen * n_batch);  // Zero out the memory
    memcpy(batched_input, input1, inlen1);  // Copy first input to the batched input
    memcpy(batched_input + inlen, input2, inlen2);  // Copy second input to the batched input

    // Allocate memory for the output
    WORD outlen = n_outbit / 8;
    BYTE *output = (BYTE *)malloc(outlen * n_batch);
    if (!output) {
        perror("Failed to allocate memory for output");
        free(input1);
        free(input2);
        free(batched_input);
        return EXIT_FAILURE;
    }

    printf("Key len: %d \n", keylen);
    
    // Perform the hashing
    START_TIMER(blake_timer)
    mcm_cuda_blake2s_hash_batch(key, keylen, batched_input, inlen, output, n_outbit, n_batch);
    END_TIMER(blake_timer, "Blake Timer")
    
    // Print the result
    printf("BLAKE2S hash (batch size = %d):\n", n_batch);
    for (WORD i = 0; i < n_batch; i++) {
        printf("Hash %d: ", i + 1);
        print_hash(output + i * outlen, outlen);
    }

    // Clean up
    free(output);
    free(input1);
    free(input2);
    free(batched_input);
    return 0;
}
