#include <stdio.h>
#include <iostream>
#include <bits/stdc++.h>
#include <cmath>
#include "ntt.cuh"
#include "../../curves/curve_config.cuh"


void generate_random_scalar_t_arr(scalar_t * arr, uint32_t n, uint32_t limbs){
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < limbs; j++) {
      arr[i].limbs_storage.limbs[j] = i;
    }
  }
}

uint32_t test_eq(scalar_t * a, scalar_t * b, uint32_t n) {
  int eq = 1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 8; j++) {
      if (b[i].limbs_storage.limbs[j] != a[i].limbs_storage.limbs[j]) eq = 0;
    }
  }
  return eq;
}

void print_array_of_scalars(scalar_t * a, uint32_t n) {
  for (int i = 0; i < n; i++) {
    std::cout << i << " ";
    for (int j = 0; j < 8; j++) {
      std::cout << a[i].limbs_storage.limbs[j] << " ";
    }
    std::cout << " " << std::endl;
  }
}

void test_ntt(int size, scalar_t * twiddles, scalar_t * twiddles_inv, uint32_t n_twiddles) {
  printf("test scalar ntt, size: %d ", size);
  uint32_t n = size;
  scalar_t * inputs = new scalar_t[n];
  generate_random_scalar_t_arr(inputs, n, 8);
  scalar_t * res = ntt(inputs, n, twiddles, n_twiddles, false);
  scalar_t * inputs_again = ntt(res, n, twiddles_inv, n_twiddles, true);
  // test_eq(res, inputs_again, n) ^ 1 checks that the ntt result was different than the input. 
  printf(" did something? %d Ok? %d \n", test_eq(res, inputs_again, n) ^ 1, test_eq(inputs, inputs_again, n));
}


void test_ntt_end2end(int size) {
  printf("test scalar ntt, size: %d ", size);
  uint32_t n = size;
  scalar_t * inputs = new scalar_t[n];
  generate_random_scalar_t_arr(inputs, n, 8);
  print_array_of_scalars(inputs,n);
  int res1 = ntt_end2end(inputs, n, false);
  int res2 = ntt_end2end(inputs, n, true);
}

int main(int argc, char * argv[]) {
  // Generate twiddle factors up to 4096
  uint32_t n_twiddles = 4096;
  scalar_t * twiddles = new scalar_t[n_twiddles];
  scalar_t * twiddles_inv = new scalar_t[n_twiddles];
  fill_twiddle_factors_array(twiddles, n_twiddles, scalar_t::omega());
  fill_twiddle_factors_array(twiddles_inv, n_twiddles, scalar_t::omega_inv());
  // Get device allocated pointes to the twiddle factors array.
  scalar_t * d_twiddles = copy_twiddle_factors_to_device(twiddles, n_twiddles);
  scalar_t * d_twiddles_inv = copy_twiddle_factors_to_device(twiddles_inv, n_twiddles);
  // Compute NTT. 
  test_ntt(4, d_twiddles, d_twiddles_inv, n_twiddles);
  test_ntt(256, twiddles, twiddles_inv, n_twiddles);
  test_ntt(512, twiddles, twiddles_inv, n_twiddles);
  test_ntt(4096, twiddles, twiddles_inv, n_twiddles);
  // Free twiddle factors. 
  cudaFree(d_twiddles);
  cudaFree(d_twiddles_inv);
  // end2end test
  test_ntt_end2end(4);
  test_ntt_end2end(256);
  test_ntt_end2end(512);
}