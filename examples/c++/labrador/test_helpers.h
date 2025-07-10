#pragma once

#include "types.h"
#include "utils.h"

constexpr bool TESTING = true;

/// print a Zq vector
void print_vec(const Zq* vec, size_t len, const std::string& name);

/// print a polynomial
void print_poly(const PolyRing& poly, const std::string& name);

/// @brief Compares two vectors of PolyRing polynomials element-wise
/// @param vec1 First vector to compare
/// @param vec2 Second vector to compare
/// @param size Number of PolyRing elements to compare
/// @return true if vectors are equal, false otherwise
bool poly_vec_eq(const PolyRing* vec1, const PolyRing* vec2, size_t size);

// Generate a random polynomial vector with coefficients bounded by max_value
std::vector<PolyRing> rand_poly_vec(size_t size, int64_t max_value);

// Generate a random EqualityInstance satisfied by the given witness S
EqualityInstance create_rand_eq_inst(size_t n, size_t r, const std::vector<Rq>& S);

// Generate a random ConstZeroInstance satisfied by the given witness S
ConstZeroInstance create_rand_const_zero_inst(size_t n, size_t r, const std::vector<Rq>& S);

// Check if the given EqualityInstance is satisfied by the witness S or not
bool witness_legit_eq(const EqualityInstance& eq_inst, const std::vector<Rq>& S);

// Check if the given EqualityInstance is satisfied by the witness S_hat (NTT form) or not
bool witness_legit_eq_all_ntt(const EqualityInstance& eq_inst, const std::vector<Tq>& S_hat);

// Check if the given ConstZeroInstance is satisfied by the witness S_hat (NTT form) or not
bool witness_legit_const_zero_all_ntt(const ConstZeroInstance& cz_inst, const std::vector<Tq>& S_hat);

// Check if the given ConstZeroInstance is satisfied by the witness S or not
bool witness_legit_const_zero(const ConstZeroInstance& cz_inst, const std::vector<Rq>& S);

bool lab_witness_legit(const LabradorInstance& lab_inst, const std::vector<Rq>& S);

void test_jl();