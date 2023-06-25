#include <iostream>
#include <chrono>
#include <vector>
#include "msm.cu"
#include "../../utils/cuda_utils.cuh"
#include "../../primitives/projective.cuh"
#include "../../primitives/field.cuh"
#include "../../curves/bls12_377/curve_config.cuh"
// #include "../../curves/bn254/curve_config.cuh"

using namespace BLS12_377;

class Dummy_Scalar {
  public:
    static constexpr unsigned NBITS = 32;

    unsigned x;
    unsigned p = 1<<31;

    friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Scalar& scalar) {
      os << scalar.x;
      return os;
    }

    HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) {
      return (x>>(digit_num*digit_width))&((1<<digit_width)-1);
    }

    friend HOST_DEVICE_INLINE Dummy_Scalar operator+(Dummy_Scalar p1, const Dummy_Scalar& p2) {   
      return {(p1.x+p2.x)%p1.p};
    }

    friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const Dummy_Scalar& p2) {
      return (p1.x == p2.x);
    }

    friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const unsigned p2) {
      return (p1.x == p2);
    }

    static HOST_DEVICE_INLINE Dummy_Scalar neg(const Dummy_Scalar &scalar) { 
      return {scalar.p-scalar.x};
    }
    static HOST_INLINE Dummy_Scalar rand_host() {
      return {(unsigned)rand()};
    }
};

class Dummy_Projective {

  public:
    Dummy_Scalar x;

    static HOST_DEVICE_INLINE Dummy_Projective zero() {
      return {0};
    }

    static HOST_DEVICE_INLINE Dummy_Projective one() {
      return {1};
    }

    static HOST_DEVICE_INLINE Dummy_Projective to_affine(const Dummy_Projective &point) {
      return {point.x};
    }

    static HOST_DEVICE_INLINE Dummy_Projective from_affine(const Dummy_Projective &point) {
      return {point.x};
    }

    static HOST_DEVICE_INLINE Dummy_Projective neg(const Dummy_Projective &point) { 
      return {Dummy_Scalar::neg(point.x)};
    }

    friend HOST_DEVICE_INLINE Dummy_Projective operator+(Dummy_Projective p1, const Dummy_Projective& p2) {   
      return {p1.x+p2.x};
    }

    // friend HOST_DEVICE_INLINE Dummy_Projective operator-(Dummy_Projective p1, const Dummy_Projective& p2) {   
    //   return p1 + neg(p2);
    // }

    friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Projective& point) {
      os << point.x;
      return os;
    }

    friend HOST_DEVICE_INLINE Dummy_Projective operator*(Dummy_Scalar scalar, const Dummy_Projective& point) {   
      Dummy_Projective res = zero();
  #ifdef CUDA_ARCH
  #pragma unroll
  #endif
      for (int i = 0; i < Dummy_Scalar::NBITS; i++) {
        if (i > 0) {
          res = res + res;
        }
        if (scalar.get_scalar_digit(Dummy_Scalar::NBITS - i - 1, 1)) {
          res = res + point;
        }
      }
      return res;
    }

    friend HOST_DEVICE_INLINE bool operator==(const Dummy_Projective& p1, const Dummy_Projective& p2) {
      return (p1.x == p2.x);
    }

    static HOST_DEVICE_INLINE bool is_zero(const Dummy_Projective &point) {
      return point.x == 0;
    }

    static HOST_INLINE Dummy_Projective rand_host() {
      return {(unsigned)rand()};
    }
};

//switch between dummy and real:

typedef scalar_t test_scalar;
typedef projective_t test_projective;
typedef affine_t test_affine;

// typedef Dummy_Scalar test_scalar;
// typedef Dummy_Projective test_projective;
// typedef Dummy_Projective test_affine;

int main()
{
  unsigned batch_size = 1;
  unsigned msm_size = 1<<24;
  unsigned N = batch_size*msm_size;

  test_scalar *scalars = new test_scalar[N];
  test_affine *points = new test_affine[N];
  
  for (unsigned i=0;i<N;i++){
    // scalars[i] = (i%msm_size < 10)? test_scalar::rand_host() : scalars[i-10];
    points[i] = (i%msm_size < 10)? test_projective::to_affine(test_projective::rand_host()): points[i-10];
    scalars[i] = test_scalar::rand_host();
    // points[i] = test_projective::to_affine(test_projective::rand_host());
  }
  std::cout<<"finished generating"<<std::endl;

  // projective_t *short_res = (projective_t*)malloc(sizeof(projective_t));
  // test_projective *large_res = (test_projective*)malloc(sizeof(test_projective));
  test_projective large_res[batch_size];
  test_projective batched_large_res[batch_size];
  // fake_point *large_res = (fake_point*)malloc(sizeof(fake_point));
  // fake_point batched_large_res[256];


  // short_msm<scalar_t, projective_t, affine_t>(scalars, points, N, short_res);
  // for (unsigned i=0;i<batch_size;i++){
    // large_msm<test_scalar, test_projective, test_affine>(scalars+msm_size*i, points+msm_size*i, msm_size, large_res+i, false);
    // std::cout<<"final result large"<<std::endl;
    // std::cout<<test_projective::to_affine(*large_res)<<std::endl;
  // }
  auto begin = std::chrono::high_resolution_clock::now();
  // batched_large_msm<test_scalar, test_projective, test_affine>(scalars, points, batch_size, msm_size, batched_large_res, false);
  large_msm<test_scalar, test_projective, test_affine>(scalars, points, msm_size, large_res, false, true);
  std::cout<<test_projective::to_affine(large_res[0])<<std::endl;
  large_msm<test_scalar, test_projective, test_affine>(scalars, points, msm_size, large_res, false, false);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
  std::cout<<test_projective::to_affine(large_res[0])<<std::endl;

  // reference_msm<test_affine, test_scalar, test_projective>(scalars, points, msm_size);

  // std::cout<<"final results batched large"<<std::endl;
  // bool success = true;
  // for (unsigned i = 0; i < batch_size; i++)
  // {
  //   std::cout<<test_projective::to_affine(batched_large_res[i])<<std::endl;
  //   if (test_projective::to_affine(large_res[i])==test_projective::to_affine(batched_large_res[i])){
  //     std::cout<<"good"<<std::endl;
  //   }
  //   else{
  //     std::cout<<"miss"<<std::endl;
  //     std::cout<<test_projective::to_affine(large_res[i])<<std::endl;
  //     success = false;
  //   }
  // }
  // if (success){
  //   std::cout<<"success!"<<std::endl;
  // }
  
  // std::cout<<batched_large_res[0]<<std::endl;
  // std::cout<<batched_large_res[1]<<std::endl;
  // std::cout<<projective_t::to_affine(batched_large_res[0])<<std::endl;
  // std::cout<<projective_t::to_affine(batched_large_res[1])<<std::endl;

  // std::cout<<"final result short"<<std::endl;
  // std::cout<<pr<<std::endl;

  return 0;
}