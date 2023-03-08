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

void generate_projective_t_array(projective_t * arr, uint32_t n){
  for (int i = 0; i < n; i++) {
    if (i % 5 == 0){
      arr[i] = {
        {3676489403, 4214943754, 4185529071, 1817569343, 387689560, 2706258495, 2541009157, 3278408783, 1336519695, 647324556, 832034708, 401724327},
        {1187375073, 212476713, 2726857444, 3493644100, 738505709, 14358731, 3587181302, 4243972245, 1948093156, 2694721773, 3819610353, 146011265},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      };
    }
    if (i % 5 == 1){
      arr[i] = {
        {927572523, 3123811016, 1178179586, 448270957, 3269025417, 873655910, 946685814, 846160237, 2311665546, 894701547, 1123227996, 414748152},
        {2100670837, 1657590303, 4206131811, 3111559769, 3261363570, 430821050, 2016803245, 2664358421, 3132350727, 189414955, 1844185218, 11036570},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      };
    } 
    if (i % 5 == 2){
      arr[i] = {
        {606938594, 3862011666, 3396180143, 765820065, 3281167117, 634141057, 210831039, 670764991, 3442481388, 2417967610, 1382165347, 243748907},
        {2486871565, 3199940895, 3186416593, 2451721591, 4108712975, 2604984942, 1165376591, 854454192, 1479545654, 1006124383, 1570319433, 22366661},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      };
    } if (i % 5 == 3){
      arr[i] = {
        {183039612, 256454025, 4250922080, 2485970688, 3679755773, 1397028634, 1298805238, 3413182507, 2291846949, 1280816489, 1119750210, 122833203},
        {3025851512, 1147574033, 1323495323, 569405769, 382481561, 1330634004, 3879950484, 1158208050, 2740575984, 2745897444, 3101936482, 405605297},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      };
    }
    if (i % 5 == 4){
      arr[i] = {
        {4006417784, 3580973450, 2524244405, 3414509667, 4142213295, 3876406748, 4116037682, 877187559, 3606672288, 3459819278, 3198860768, 30571621},
        {182896763, 2741166359, 626891178, 1601768019, 1967793394, 706302600, 2612369182, 2051460370, 2918333441, 1902350841, 475238909, 239719017},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
      };
    }
  }
}

uint32_t test_eq_ec(affine_t * a, affine_t * b, uint32_t n) {
  int eq = 1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 12; j++) {
      if (b[i].x.limbs_storage.limbs[j] != a[i].x.limbs_storage.limbs[j]) eq = 0;
      if (b[i].y.limbs_storage.limbs[j] != a[i].y.limbs_storage.limbs[j]) eq = 0;
    }
  }
  return eq;
}

uint32_t test_eq_ec_proj(projective_t * a, projective_t * b, uint32_t n) {
  int eq = 1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < 12; j++) {
      if (b[i].x.limbs_storage.limbs[j] != a[i].x.limbs_storage.limbs[j]) eq = 0;
      if (b[i].y.limbs_storage.limbs[j] != a[i].y.limbs_storage.limbs[j]) eq = 0;
      if (b[i].z.limbs_storage.limbs[j] != a[i].z.limbs_storage.limbs[j]) eq = 0;
    }
  }
  return eq;
}

__global__ void to_affine_kernel(projective_t * a, affine_t * res, uint32_t n) {
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
  res[tid] = projective_t::to_affine(a[tid]);
}

affine_t * to_affine(projective_t * arr, uint32_t n) {
  size_t size_proj = n * sizeof(projective_t);
  size_t size_aff = n * sizeof(affine_t);
  affine_t * res = new affine_t[n];
  affine_t * d_res;
  cudaMalloc( & d_res, size_aff);
  projective_t * d_arr;
  cudaMalloc( & d_arr, size_proj);
  cudaMemcpy(d_arr, arr, size_proj, cudaMemcpyHostToDevice);
  to_affine_kernel << < 1, n >>> (d_arr, d_res, n);
  cudaDeviceSynchronize();
  cudaMemcpy(res, d_res, size_aff, cudaMemcpyDeviceToHost);
  cudaFree(d_res);
  cudaFree(d_arr);
  return res;
}

void test_ec_ntt(int size, scalar_t * twiddles, scalar_t * twiddles_inv, uint32_t n_twiddles) {
  printf("test ec ntt, size: %d. OK? ", size);
  uint32_t n = size;
  projective_t * inputs = new projective_t[n];
  generate_projective_t_array(inputs, n);
  affine_t * i = to_affine(inputs, n);
  projective_t * res = ecntt(inputs, n, twiddles, n_twiddles, false);
  projective_t * inputs_again = ecntt(res, n, twiddles_inv, n_twiddles, true);
  affine_t * ia = to_affine(inputs_again, n);
  // test_eq_ec_proj(inputs, res, n) ^ 1 checks that the ntt result was different than the input. 
  printf(" did something? %d Ok? %d \n", test_eq_ec_proj(inputs, res, n) ^ 1, test_eq_ec(i, ia, n));
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
  test_ec_ntt(4, d_twiddles, d_twiddles_inv, n_twiddles);
  test_ec_ntt(256, d_twiddles, d_twiddles_inv, n_twiddles);
  test_ec_ntt(512, d_twiddles, d_twiddles_inv, n_twiddles);
  test_ntt(4, d_twiddles, d_twiddles_inv, n_twiddles);
  test_ntt(256, twiddles, twiddles_inv, n_twiddles);
  test_ntt(512, twiddles, twiddles_inv, n_twiddles);
  test_ntt(4096, twiddles, twiddles_inv, n_twiddles);
  // Free twiddle factors. 
  cudaFree(d_twiddles);
  cudaFree(d_twiddles_inv);
}