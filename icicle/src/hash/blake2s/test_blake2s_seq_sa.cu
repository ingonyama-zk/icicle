#include <chrono>
#include "gpu-utils/device_context.cuh"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "extern.cu"
#include "hash/blake2s/blake2s.cuh"

using namespace blake2s;

#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg) \
  printf("%s: %.0f us\n", msg, FpMicroseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

extern "C" {
void mcm_cuda_blake2s_hash_batch(BYTE *key, WORD keylen, BYTE *in, WORD inlen, BYTE *out, WORD n_outbit, WORD n_batch);
}

void print_hash(BYTE *hash, WORD len) {
    printf("Hash Len: %d \n", len);
    printf("BLAKE2S hash:\n");
    for (WORD i = 0; i < len; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

std::string byte_to_hex(BYTE *data, WORD len) {
    std::stringstream ss;
    for (WORD i = 0; i < len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)data[i];
    }
    return ss.str();
}

std::vector<std::string> load_csv(const char *filename) {
    std::vector<std::string> hashes;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        // Directly add the line as a hash, assuming one hash per line
        hashes.push_back(line);
    }
    return hashes;
}




int main(int argc, char **argv) {
    using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
    using FpMicroseconds = std::chrono::duration<float, std::chrono::microseconds::period>;

    std::vector<std::string> test_strings = {
        "0", "01", "012", "0123", "01234", "012345", "0123456", "01234567", "012345678", "0123456789"
    };

    const char *csv_filename = "expected_hashes.csv";  // Replace with your actual CSV file name
    std::vector<std::string> expected_hashes = load_csv(csv_filename);
    assert(expected_hashes.size() == test_strings.size() && "Number of hashes in CSV must match number of test strings.");
    std::cout << "Loaded hashes from CSV:" << std::endl;
    // for (size_t i = 0; i < expected_hashes.size(); ++i) {
    //     std::cout << "Expected hash " << i  << ": " << expected_hashes[i] << std::endl;
    // }

    // Test parameters
    WORD n_outbit = 256;  // Output length in bits
    WORD n_batch = 1;  // Number of hashes to compute in parallel

    // Test parameters
    BYTE key[32] = "";  // Example key
    WORD keylen = strlen((char *)key);

    // Allocate memory for the output
    WORD outlen = n_outbit / 8;

    // Perform the hashing
    HashConfig config = default_hash_config();

    for (size_t i = 0; i < test_strings.size(); i++) {

        BYTE *output = (BYTE *)malloc(outlen * n_batch);
        if (!output) {
            perror("Failed to allocate memory for output");
            return EXIT_FAILURE;
        }

        const std::string &input_str = test_strings[i];
        BYTE *input = (BYTE *)input_str.c_str();
        size_t inlen = input_str.size();



        // Perform the hashing
        START_TIMER(blake_timer)
        mcm_cuda_blake2s_hash_batch(key, keylen, input, inlen, output, outlen, n_batch);
        END_TIMER(blake_timer, "Blake Timer")
        // Convert the output to hex string
        std::string computed_hash = byte_to_hex(output, outlen);
        // Compare with the expected hash

        
        if (computed_hash == expected_hashes[i]) {
            std::cout << "Test " << i  << " passed." << std::endl;
        } else {
            std::cout << "Test " << i << " failed." << std::endl;    
            std::cout << "Expected: " << expected_hashes[i] << std::endl;
            std::cout << "Got:      " << computed_hash << std::endl;
        }
    free(output);
    }
    
    return 0;
    
}

