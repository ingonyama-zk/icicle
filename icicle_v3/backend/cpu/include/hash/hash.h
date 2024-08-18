#pragma once
#ifndef HASH_H
#define HASH_H

#include <cassert>
#include <vector>
#include <stdexcept>
#include <cstdint>

/**
 * @namespace hash
 * Includes classes and methods for describing hash functions.
 */
namespace hash {

  /**
   * @struct HashConfig
   * Encodes hash operations parameters.
   */
  struct HashConfig {
    bool is_async; /**< Whether to run the hash operations asynchronously. In this CPU version, it's not used. */
  };

  /**
   * A function that returns the default value of HashConfig.
   * @return Default value of HashConfig.
   */
  static HashConfig default_hash_config()
  {
    HashConfig config = {
      false, // is_async
    };
    return config;
  }

  /**
   * @class Hasher
   *
   * An interface containing methods for hashing
   *
   * @tparam PreImage type of inputs elements
   * @tparam Image type of state elements. Also used to describe the type of hash output
   */
  template <typename PreImage, typename Image>
  class Hasher
  {
  public:
    /// @brief the width of permutation state
    const unsigned int width;

    /// @brief how many elements a state can fit per 1 permutation. Used with domain separation.
    const unsigned int preimage_max_length;

    /// @brief portion of the state to absorb input into, or squeeze output from
    const unsigned int rate;

    /// @brief start squeezing from this offset. Used with domain separation.
    const unsigned int offset;

    Hasher(unsigned int width, unsigned int preimage_max_length, unsigned int rate, unsigned int offset)
        : width(width), preimage_max_length(preimage_max_length), rate(rate), offset(offset)
    {
      assert(
        rate * sizeof(PreImage) <= preimage_max_length * sizeof(Image) &&
        "Input rate can not be bigger than preimage max length");
    }

    virtual void hash_2d(
      const std::vector<std::vector<PreImage>>* inputs,
      Image* states,
      unsigned int number_of_inputs,
      unsigned int output_len,
      uint64_t number_of_rows) const
    {
      throw std::runtime_error("Absorb 2d is not implemented for this hash");
    }

    virtual void compress_and_inject(
      const std::vector<std::vector<PreImage>>* matrices_to_inject,
      unsigned int number_of_inputs,
      uint64_t number_of_rows,
      const Image* prev_layer,
      Image* next_layer,
      unsigned int digest_elements) const
    {
      throw std::runtime_error("Compress and inject is not implemented for this hash");
    }

    void compress_many(
      const PreImage* input,
      Image* output,
      unsigned int number_of_states,
      unsigned int output_len,
      const HashConfig& cfg) const
    {
      hash_many((const PreImage*)(input), output, number_of_states, width, output_len, cfg);
    }

    // virtual void run_hash_many_kernel(
    //   const std::vector<PreImage>& input,
    //   std::vector<Image>& output,
    //   unsigned int number_of_states,
    //   unsigned int input_len,
    //   unsigned int output_len) const
    // {
    //   throw std::runtime_error("Hash many kernel is not implemented for this hash");
    // }

    virtual void run_hash(const PreImage* input, Image* output, size_t input_len, size_t output_len, const HashConfig& cfg) const
    {
      throw std::runtime_error("Hash many kernel is not implemented for this hash");
    }

    void run_hash_many(
      const PreImage* input,
      Image* output,
      unsigned int batch_size,
      size_t input_len,
      size_t output_len,
      const HashConfig& cfg) const
    {
      for (unsigned int i = 0; i < batch_size; ++i) {
        // Call run_hash for each batch
        run_hash(input, output, input_len, output_len, cfg);

        // Move the input pointer forward by the size of the input data for one batch
        input += input_len * sizeof(PreImage);

        // Move the output pointer forward by the size of the output data for one batch
        output += output_len * sizeof(Image);
      }
    }
  };
} // namespace hash

#endif