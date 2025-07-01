#include "oracle.h"

std::vector<std::byte> Oracle::generate(const std::byte* msg, size_t len)
{
  // buffer = state || msg
  std::vector<std::byte> buf;
  buf.reserve(state_.size() + len);
  buf.insert(buf.end(), state_.begin(), state_.end());
  buf.insert(buf.end(), msg, msg + len);

  std::vector<std::byte> digest(hasher_.output_size());
  hasher_.hash(buf.data(), buf.size(), {}, digest.data());

  state_ = digest; // update internal state
  return digest;   // return challenge
}