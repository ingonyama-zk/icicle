#include <iostream>
#include <chrono>
#include <vector>
#include "msm.cu"
#include "../../utils/cuda_utils.cuh"
#include "../../primitives/projective.cuh"
#include "../../primitives/field.cuh"
#include "../../curves/curve_config.cuh"


struct fake_point
{
  unsigned val = 0;

  __host__ __device__ inline fake_point operator+(fake_point fp) {
        return {val+fp.val};
    }

  __host__ __device__ fake_point zero() {
        fake_point p;
        return p;
    }

};

std::ostream& operator<<(std::ostream &strm, const fake_point &a) {
  return strm <<a.val;
}

struct fake_scalar
{
  unsigned val = 0;
  unsigned bitsize = 32;

  // __host__ __device__ unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width){

  //   return (val>>(digit_num*digit_width))&((1<<digit_width)-1);

  // }
  __host__ __device__ int get_scalar_digit(int digit_num, int digit_width){

    return (val>>(digit_num*digit_width))&((1<<digit_width)-1);

  }

  __host__ __device__ inline fake_point operator*(fake_point fp) {
      
      fake_point p1;
      fake_point p2;
      unsigned x = val;
      if (x == 0) return fake_point().zero();

      unsigned i = 1;
      unsigned c_bit = (x & (1<<(bitsize-1)))>>(bitsize-1);
      while (c_bit==0 && i<bitsize){
        i++;
        c_bit = (x & (1<<(bitsize-i)))>>(bitsize-i);
      }
      p1 = fp;
      p2 = p1+p1;
      while (i<bitsize){
        i++;
        c_bit = (x & (1<<(bitsize-i)))>>(bitsize-i);
        if (c_bit){
          p1 = p1 + p2;
          p2 = p2 + p2;
        }
        else {
          p2 = p1 + p2;
          p1 = p1 + p1;
        }
      }
      
      return p1;
  }

};

class Dummy_Scalar {
  public:
    static constexpr unsigned NBITS = 32;

    unsigned x;

    friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Dummy_Scalar& scalar) {
      os << scalar.x;
      return os;
    }

    HOST_DEVICE_INLINE unsigned get_scalar_digit(unsigned digit_num, unsigned digit_width) {
      return (x>>(digit_num*digit_width))&((1<<digit_width)-1);
    }

    friend HOST_DEVICE_INLINE Dummy_Scalar operator+(Dummy_Scalar p1, const Dummy_Scalar& p2) {   
      return {p1.x+p2.x};
    }

    friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const Dummy_Scalar& p2) {
      return (p1.x == p2.x);
    }

    friend HOST_DEVICE_INLINE bool operator==(const Dummy_Scalar& p1, const unsigned p2) {
      return (p1.x == p2);
    }

    // static HOST_DEVICE_INLINE Dummy_Scalar neg(const Dummy_Scalar &scalar) { 
    //   return {Dummy_Scalar::neg(point.x)};
    // }
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

    static HOST_DEVICE_INLINE Dummy_Projective to_affine(const Dummy_Projective &point) {
      return {point.x};
    }

    static HOST_DEVICE_INLINE Dummy_Projective from_affine(const Dummy_Projective &point) {
      return {point.x};
    }

    // static HOST_DEVICE_INLINE Dummy_Projective neg(const Dummy_Projective &point) { 
    //   return {Dummy_Scalar::neg(point.x)};
    // }

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

// typedef scalar_t test_scalar;
// typedef projective_t test_projective;
// typedef affine_t test_affine;

typedef Dummy_Scalar test_scalar;
typedef Dummy_Projective test_projective;
typedef Dummy_Projective test_affine;

int main()
{
  // fake_point p1;
  // fake_point p2;
  // p1.val = 8;
  // p2.val = 9;
  // std::cout<<(p1+p2)<<std::endl;

  // unsigned N = 4;
  // unsigned batch_size = 1<<0;
  unsigned batch_size = 1;
  unsigned msm_size = 1<<26;
  unsigned N = batch_size*msm_size;

  //1<<2, 1<<4 - gets stuck..? V but not for c=10 with real
  //1<<2, 1<<12 - first results for all.. 
  //1<<3, 1<<6 - first for all  VV
  //1<<2, 1<<6 - wrong results for all V but not for c=10 with real

  //c==10
  //3,4 - stuck V
  //2,7 - wrong
  //2,6 - wrong V
  //2,10 - wrong
  //2,11 - good
  //2,12 - good
  //3,6 - good
  //3,5 - good
  //4,4 - good
  //4,3 - good
  //8,12 - good V

  // fake_scalar scalars[N];
  // fake_point points[N];

  test_scalar *scalars = new test_scalar[N];
  test_affine *points = new test_affine[N];
  
  // std::vector<scalar_t> scalars;
  // std::vector<affine_t> points;
  // scalars.reserve(N);
  // points.reserve(N);

  // srand(time(NULL));
  // for (unsigned i = 0; i < N; i++)
  // {
  //   // scalars[i].val = rand()%(1<<10);
  //   scalars[i] = {unsigned(rand()%(1<<10))};
  //   // std::cout<<scalars[i].val<<std::endl;
  //   // points[i].val = rand()%(1<<10);
  //   points[i] = {{unsigned(rand()%(1<<10)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  //             {unsigned(rand()%(1<<10)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  //   // std::cout<<points[i].val<<std::endl;
  // }

  // scalars[0] = {3827484040, 2874625294, 4134484179, 2522098891, 1684039639, 4190761864, 1674792009, 1733172596};
  // scalars[1] = {2859778174, 247198543, 1069683537, 986951671, 18230349, 1865405355, 3834758898, 1605705230};
  // scalars[2] = {870702623, 2140804196, 1118323047, 4097847923, 733285949, 2517599604, 2748585063, 1465198310};
  // scalars[3] = {2997668077, 4130616472, 4255624276, 3713720096, 813455961, 1818410993, 1796699074, 1289452986};

  // points[0].x = {927572523, 3123811016, 1178179586, 448270957, 3269025417, 873655910, 946685814, 846160237, 2311665546, 894701547, 1123227996, 414748152};
  // points[0].y = {2100670837, 1657590303, 4206131811, 3111559769, 3261363570, 430821050, 2016803245, 2664358421, 3132350727, 189414955, 1844185218, 11036570};
  // points[1].x = {606938594, 3862011666, 3396180143, 765820065, 3281167117, 634141057, 210831039, 670764991, 3442481388, 2417967610, 1382165347, 243748907};
  // points[1].y = {2486871565, 3199940895, 3186416593, 2451721591, 4108712975, 2604984942, 1165376591, 854454192, 1479545654, 1006124383, 1570319433, 22366661};
  // points[2].x = {183039612, 256454025, 4250922080, 2485970688, 3679755773, 1397028634, 1298805238, 3413182507, 2291846949, 1280816489, 1119750210, 122833203};
  // points[2].y = {3025851512, 1147574033, 1323495323, 569405769, 382481561, 1330634004, 3879950484, 1158208050, 2740575984, 2745897444, 3101936482, 405605297};
  // points[3].x = {4006417784, 3580973450, 2524244405, 3414509667, 4142213295, 3876406748, 4116037682, 877187559, 3606672288, 3459819278, 3198860768, 30571621};
  // points[3].y = {182896763, 2741166359, 626891178, 1601768019, 1967793394, 706302600, 2612369182, 2051460370, 2918333441, 1902350841, 475238909, 239719017};

  srand(11);

  for (unsigned i=0;i<N;i++){
    scalars[i] = (i%msm_size < 10)? test_scalar::rand_host() : scalars[i-10];
    points[i] = (i%msm_size < 10)? test_projective::to_affine(test_projective::rand_host()): points[i-10];
  }
  std::cout<<"finished generating"<<std::endl;

  // scalars[0] = scalar_t::rand_host();
  // scalars[1] = scalar_t::rand_host();
  // scalars[2] = scalar_t::rand_host();
  // scalars[3] = scalar_t::rand_host();

  // points[0] = projective_t::to_affine(projective_t::rand_host());
  // points[1] = projective_t::to_affine(projective_t::rand_host());
  // points[2] = projective_t::to_affine(projective_t::rand_host());
  // points[3] = projective_t::to_affine(projective_t::rand_host());
/*correct result:
1557917178, 269077943, 1116505460, 728110787, 4176849812, 3140203189, 2756051319, 197704154, 1838744007, 2201658078, 1505047534, 239949230, 
2029063365, 2557489072, 3905272471, 2418563649, 2077595491, 357415053, 3188715161, 1890916285, 354886608, 410171932, 1437862573, 206970588, 
4160033405, 2697065480, 1940009895, 2097886176, 4019146882, 2931880476, 3425684730, 2783686325, 1918054479, 1505257125, 3268347217, 269536830, */

// scalars[0].val = 456;
//   scalars[1].val = 51;
//   scalars[2].val = 984;
//   scalars[3].val = 15;

//   points[0].val = 0;
//   points[1].val = 1;
//   points[2].val = 2;
//   points[3].val = 3;

  // for (unsigned i = 1; i < N/4; i++)
  // {
  //   scalars[4*i+0] = scalars[0];
  //   scalars[4*i+1] = scalars[1];
  //   scalars[4*i+2] = scalars[2];
  //   scalars[4*i+3] = scalars[3];
  //   points[4*i+0] = points[0];
  //   points[4*i+1] = points[1];
  //   points[4*i+2] = points[2];
  //   points[4*i+3] = points[3];
  // }
  

  // std::cout<<"scalars"<<std::endl;
  // for (unsigned j = 0; j<N ; j++){

  // for (unsigned i = 0; i < 8; i++) {
  //   std::cout << scalars[j].limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";

  // }

  // std::cout<<"points"<<std::endl;
  // for (unsigned j = 0; j<N ; j++){
  // for (unsigned i = 0; i < 12; i++) {
  //   std::cout << points[j].x.limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";
  // for (unsigned i = 0; i < 12; i++) {
  //   std::cout << points[j].y.limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";

  // // return 0;
  // // for (unsigned i = 0; i < 8; i++) {
  // //   std::cout << points[j].z.limbs_storage.limbs[i] << ", ";
  // // }
  // // std::cout << "\n";
  // }
  
  // projective_t test_p = projective_t::zero();

  // for (unsigned i = 0; i < 8; i++) {
  //   std::cout << test_p.x.limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";
  // for (unsigned i = 0; i < 8; i++) {
  //   std::cout << test_p.y.limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";
  // for (unsigned i = 0; i < 8; i++) {
  //   std::cout << test_p.z.limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";

  // projective_t test_mul = scalars[0]*points[0];

  // for (unsigned i = 0; i < 8; i++) {
  //   std::cout << test_mul.x.limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";
  // for (unsigned i = 0; i < 8; i++) {
  //   std::cout << test_mul.y.limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";
  // for (unsigned i = 0; i < 8; i++) {
  //   std::cout << test_mul.z.limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";


  // cuda_ctx my_ctx = cuda_ctx(0);
  // large_msm<fake_point, fake_scalar>(my_ctx, points, scalars, N);

  // bucket_method_msm<scalar_t,projective_t, affine_t>(255, 10, scalars, points, N);
  // projective_t pr=short_msm<scalar_t,projective_t,affine_t>(scalars, points, N);

  // projective_t *short_res = (projective_t*)malloc(sizeof(projective_t));
  // test_projective *large_res = (test_projective*)malloc(sizeof(test_projective));
  test_projective large_res[batch_size];
  test_projective batched_large_res[batch_size];
  // fake_point *large_res = (fake_point*)malloc(sizeof(fake_point));
  // fake_point batched_large_res[256];

  // projective_t short_res[1];
  // projective_t large_res[1];

  // short_msm<scalar_t, projective_t, affine_t>(scalars, points, N, short_res);
  // for (unsigned i=0;i<batch_size;i++){
  //   large_msm<test_scalar, test_projective, test_affine>(scalars+msm_size*i, points+msm_size*i, msm_size, large_res+i);
  //   // std::cout<<"final result large"<<std::endl;
  //   // std::cout<<test_projective::to_affine(*large_res)<<std::endl;
  // }
  auto begin = std::chrono::high_resolution_clock::now();
  // batched_large_msm<test_scalar, test_projective, test_affine>(scalars, points, batch_size, msm_size, batched_large_res, false);
  large_msm<test_scalar, test_projective, test_affine>(scalars, points, msm_size, large_res, false);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
  printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
  std::cout<<test_projective::to_affine(large_res[0])<<std::endl;
  // large_msm<fake_scalar, fake_point, fake_point>(scalars, points, 4, large_res);
  // batched_large_msm<fake_scalar, fake_point, fake_point>(scalars, points, 256, 4, batched_large_res);

  // std::cout<<"final result short"<<std::endl;
  // std::cout<<*short_res<<std::endl;
  // std::cout<<"final result large"<<std::endl;
  // std::cout<<projective_t::to_affine(*large_res)<<std::endl;
  // std::cout<<*large_res<<std::endl;
  // std::cout<<*large_res<<std::endl;
  reference_msm<test_affine, test_scalar, test_projective>(scalars, points, msm_size);

  // std::cout<<"final results batched large"<<std::endl;
  // std::cout<<projective_t::to_affine(batched_large_res[0])<<std::endl;
  // bool success = true;
  // for (unsigned i = 0; i < batch_size; i++)
  // {
    // std::cout<<test_projective::to_affine(batched_large_res[i])<<std::endl;
    // if (test_projective::to_affine(large_res[i])==test_projective::to_affine(batched_large_res[i])){
    //   std::cout<<"good"<<std::endl;
    // }
    // else{
    //   std::cout<<"miss"<<std::endl;
    //   std::cout<<test_projective::to_affine(large_res[i])<<std::endl;
    //   success = false;
    // }
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