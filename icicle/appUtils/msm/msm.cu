#include <iostream>
#include <vector>
#include <cub/device/device_radix_sort.cuh>
#include "../../utils/cuda_utils.cuh"
#include "../../primitives/projective.cuh"
#include "../../primitives/base_curve.cuh"
#include "../../curves/curve_config.cuh"


#define BIG_TRIANGLE
// #define SSM_SUM

//this kernel performs single scalar multiplication
//each thread multilies a single scalar and point
template <typename P, typename S>
__global__ void ssm_kernel(S *scalars, P *points, P *results, unsigned N) {

  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) results[tid] = scalars[tid]*points[tid];

}

//this kernel sums all the elements in a given vector using multiple threads
template <typename P>
__global__ void sum_reduction_kernel(P *v, P* v_r) {

	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Start at 1/2 block stride and divide by two each iteration
	for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			v[tid] = v[tid] + v[tid + s];
		}
	}

	// Let the thread 0 for this block write the final result
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = v[tid];
	}
}

//this kernel initializes the buckets with zero points
//each thread initializes a different bucket
template <typename P>
__global__ void initialize_buckets_kernel(P *buckets, unsigned N) {
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) buckets[tid] = P::zero(); //zero point

}

//this kernel splits the scalars into digits of size c
//each thread splits a single scalar into nof_bms digits
template <typename S>
__global__ void split_scalars_kernel(unsigned *buckets_indices, unsigned *point_indices, S *scalars, unsigned problem_size, unsigned nof_bms, unsigned c){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned bucket_index;
  unsigned current_index;
  if (tid < problem_size){
    S scalar = scalars[tid];

    for (unsigned bm = 0; bm < nof_bms; bm++)
    {
      bucket_index = scalar.get_scalar_digit(bm, c);
      current_index = bm * problem_size + tid;
      buckets_indices[current_index] = (bm<<c) | bucket_index;  //the bucket module number is appended at the msbs
      point_indices[current_index] = tid; //the point index is saved for later
    }
  }
}

//this kernel adds up the points in each bucket
template <typename P, typename A>
__global__ void accumulate_buckets_kernel(P *__restrict__ buckets, unsigned *__restrict__ bucket_offsets,
               unsigned *__restrict__ bucket_sizes, unsigned *__restrict__ single_bucket_indices, unsigned *__restrict__ point_indices, A *__restrict__ points, unsigned nof_buckets){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned bucket_index = single_bucket_indices[tid];
  unsigned bucket_size = bucket_sizes[tid];
  if (tid>=nof_buckets || bucket_size == 0){ //if the bucket is empty we don't need to continue
    return;
  }
  unsigned bucket_offset = bucket_offsets[tid];
  for (unsigned i = 0; i < bucket_sizes[tid]; i++)  //add the relevant points starting from the relevant offset up to the bucket size
  {
    buckets[bucket_index] = buckets[bucket_index] + points[point_indices[bucket_offset+i]];
  }
}

//this kernel sums the entire bucket module
//each thread deals with a single bucket module
template <typename P>
__global__ void big_triangle_sum_kernel(P* buckets, P* final_sums, unsigned nof_bms, unsigned c){

  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid>nof_bms) return;
  P line_sum = buckets[(tid+1)*(1<<c)-1];
  final_sums[tid] = line_sum;
  for (unsigned i = (1<<c)-2; i >0; i--)
  {
    line_sum = line_sum + buckets[tid*(1<<c) + i];  //using the running sum method
    final_sums[tid] = final_sums[tid] + line_sum;
  }
}

//this kernel uses single scalar multiplication to multiply each bucket by its index
//each thread deals with a single bucket
template <typename P, typename S>
__global__ void ssm_buckets_kernel(P* buckets, unsigned* single_bucket_indices, unsigned nof_buckets, unsigned c){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid>nof_buckets) return;
  unsigned bucket_index = single_bucket_indices[tid];
  S scalar_bucket_multiplier;
  scalar_bucket_multiplier = {bucket_index&((1<<c)-1), 0, 0, 0, 0, 0, 0, 0}; //the index without the bucket module index
  buckets[bucket_index] = scalar_bucket_multiplier*buckets[bucket_index];

}

//this kernel computes the final result using the double and add algorithm
//it is done by a single thread
template <typename P, typename S>
__global__ void final_accumulation_kernel(P* final_sums, P* final_result, unsigned nof_bms, unsigned c){
  
  *final_result = P().zero();
  S digit_base = {unsigned(1<<c)};
  for (unsigned i = nof_bms; i >0; i--)
  {
    *final_result = digit_base*(*final_result) + final_sums[i-1];
  }
  

}

//this function computes msm using the bucket method
template <typename S, typename P, typename A>
void bucket_method_msm(unsigned bitsize, unsigned c, S *h_scalars, A *h_points, unsigned size, P*h_final_result){
  
  //copy scalars and point to gpu
  S *scalars;
  A *points;

  cudaMalloc(&scalars, sizeof(S) * size);
  cudaMalloc(&points, sizeof(A) * size);
  cudaMemcpy(scalars, h_scalars, sizeof(S) * size, cudaMemcpyHostToDevice);
  cudaMemcpy(points, h_points, sizeof(A) * size, cudaMemcpyHostToDevice);

  P *buckets;
  //compute number of bucket modules and number of buckets in each module
  unsigned nof_bms = bitsize/c;
  if (bitsize%c){
    nof_bms++;
  }
  unsigned nof_buckets = nof_bms<<c;
  cudaMalloc(&buckets, sizeof(P) * nof_buckets); 

  //lanch the bucket initialization kernel with maximum threads
  unsigned NUM_THREADS = 1 << 10;
  unsigned NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, nof_buckets);

  unsigned *bucket_indices;
  unsigned *point_indices;
  cudaMalloc(&bucket_indices, sizeof(unsigned) * size * (nof_bms+1));
  cudaMalloc(&point_indices, sizeof(unsigned) * size * (nof_bms+1));

  //split scalars into digits
  NUM_THREADS = 1 << 10;
  NUM_BLOCKS = (size * (nof_bms+1) + NUM_THREADS - 1) / NUM_THREADS;
  split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(bucket_indices + size, point_indices + size, scalars, size, nof_bms, c);

  //sort indices - the indices are sorted from smallest to largest in order to group together the points that belong to each bucket
  unsigned *sort_indices_temp_storage{};
  size_t sort_indices_temp_storage_bytes;
  cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + size, bucket_indices,
                                 point_indices + size, point_indices, size);
  cudaMalloc(&sort_indices_temp_storage, sort_indices_temp_storage_bytes);
  for (unsigned i = 0; i < nof_bms; i++) {
    unsigned offset_out = i * size;
    unsigned offset_in = offset_out + size;
    cub::DeviceRadixSort::SortPairs(sort_indices_temp_storage, sort_indices_temp_storage_bytes, bucket_indices + offset_in,
                                  bucket_indices + offset_out, point_indices + offset_in, point_indices + offset_out, size);
  }
  cudaFree(sort_indices_temp_storage);

  //find bucket_sizes
  unsigned *single_bucket_indices;
  unsigned *bucket_sizes;
  unsigned *nof_buckets_to_compute;
  cudaMalloc(&single_bucket_indices, sizeof(unsigned)*nof_buckets);
  cudaMalloc(&bucket_sizes, sizeof(unsigned)*nof_buckets);
  cudaMalloc(&nof_buckets_to_compute, sizeof(unsigned));
  unsigned *encode_temp_storage{};
  size_t encode_temp_storage_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                        nof_buckets_to_compute, nof_bms*size);
  cudaMalloc(&encode_temp_storage, encode_temp_storage_bytes);
  cub::DeviceRunLengthEncode::Encode(encode_temp_storage, encode_temp_storage_bytes, bucket_indices, single_bucket_indices, bucket_sizes,
                                        nof_buckets_to_compute, nof_bms*size);
  cudaFree(encode_temp_storage);

  //get offsets - where does each new bucket begin
  unsigned* bucket_offsets;
  cudaMalloc(&bucket_offsets, sizeof(unsigned)*nof_buckets);
  unsigned* offsets_temp_storage{};
  size_t offsets_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets);
  cudaMalloc(&offsets_temp_storage, offsets_temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(offsets_temp_storage, offsets_temp_storage_bytes, bucket_sizes, bucket_offsets, nof_buckets);
  cudaFree(offsets_temp_storage);

  //launch the accumulation kernel with maximum threads
  NUM_THREADS = 1 << 8;
  NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
  accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, bucket_offsets, bucket_sizes, single_bucket_indices, point_indices, points, nof_buckets);

  #ifdef SSM_SUM
    //sum each bucket
    NUM_THREADS = 1 << 10;
    NUM_BLOCKS = (nof_buckets + NUM_THREADS - 1) / NUM_THREADS;
    ssm_buckets_kernel<fake_point, fake_scalar><<<NUM_BLOCKS, NUM_THREADS>>>(buckets, single_bucket_indices, nof_buckets, c);
   
    //sum each bucket module
    P* final_results;
    cudaMalloc(&final_results, sizeof(P) * nof_bms);
    NUM_THREADS = 1<<c;
    NUM_BLOCKS = nof_bms;
    sum_reduction_kernel<<<NUM_BLOCKS,NUM_THREADS>>>(buckets, final_results);
  #endif

  #ifdef BIG_TRIANGLE
    P* final_results;
    cudaMalloc(&final_results, sizeof(P) * nof_bms);
    //launch the bucket module sum kernel - a thread for each bucket module
    NUM_THREADS = nof_bms;
    NUM_BLOCKS = 1;
    big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(buckets, final_results, nof_buckets, c);
    
  #endif

  P* final_result;
  cudaMalloc(&final_result, sizeof(P));
  //launch the double and add kernel, a single thread
  final_accumulation_kernel<P, S><<<1,1>>>(final_results, final_result, nof_bms, c);
  
  //copy final result to host
  cudaDeviceSynchronize();
  cudaMemcpy(h_final_result, final_result, sizeof(P), cudaMemcpyDeviceToHost);

  //free memory
  cudaFree(buckets);
  cudaFree(points);
  cudaFree(scalars);
  cudaFree(bucket_indices);
  cudaFree(point_indices);
  cudaFree(single_bucket_indices);
  cudaFree(bucket_sizes);
  cudaFree(nof_buckets_to_compute);
  cudaFree(bucket_offsets);
  cudaFree(final_results);
  cudaFree(final_result);

}

//this kernel converts affine points to projective points
//each thread deals with a single point
template <typename P, typename A>
__global__ void to_proj_kernel(A* affine_points, P* proj_points, unsigned N){
  
  unsigned tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) proj_points[tid] = P::from_affine(affine_points[tid]);
}

//the function computes msm using ssm
template <typename S, typename P, typename A>
void short_msm(S *h_scalars, A *h_points, unsigned size, P* h_final_result){ //works up to 2^10
  
  S *scalars;
  A *a_points;
  P *p_points;
  P *results;

  cudaMalloc(&scalars, sizeof(S) * size);
  cudaMalloc(&a_points, sizeof(A) * size);
  cudaMalloc(&p_points, sizeof(P) * size);
  cudaMalloc(&results, sizeof(P) * size);

  //copy inputs to device
  cudaMemcpy(scalars, h_scalars, sizeof(S) * size, cudaMemcpyHostToDevice);
  cudaMemcpy(a_points, h_points, sizeof(A) * size, cudaMemcpyHostToDevice);

  //convert to projective representation and multiply each point by its scalar using single scalar multiplication
  unsigned NUM_THREADS = size;
  to_proj_kernel<<<1,NUM_THREADS>>>(a_points, p_points, size);
  ssm_kernel<<<1,NUM_THREADS>>>(scalars, p_points, results, size);

  P *final_result;
  cudaMalloc(&final_result, sizeof(P));

  //assuming msm size is a power of 2
  //sum all the ssm results
  NUM_THREADS = size;
  sum_reduction_kernel<<<1,NUM_THREADS>>>(results, final_result);

  //copy result to host
  cudaDeviceSynchronize();
  cudaMemcpy(h_final_result, final_result, sizeof(P), cudaMemcpyDeviceToHost);

  //free memory
  cudaFree(scalars);
  cudaFree(a_points);
  cudaFree(p_points);
  cudaFree(results);
  cudaFree(final_result);

}

//the function computes msm on the host using the naive method
template <typename P, typename S>
void reference_msm(S* scalars, P* points, unsigned size){
  P res = P::zero();
  
  for (unsigned i = 0; i < size; i++)
  {
    res = res + scalars[i]*points[i];
  }

  std::cout<<"reference results"<<std::endl;
  std::cout<<res<<std::endl;
  
}

//this function is used to compute msms of size larger than 1024
template <typename S, typename P, typename A>
void large_msm(S* scalars, A* points, unsigned size, P* result){
  unsigned c = 10;
  unsigned bitsize = 255;
  bucket_method_msm(bitsize, c, scalars, points, size, result);
}

int main()
{
  // fake_point p1;
  // fake_point p2;
  // p1.val = 8;
  // p2.val = 9;
  // std::cout<<(p1+p2)<<std::endl;

  unsigned N = 4;
  // unsigned N = 1<<12;

  // fake_scalar scalars[N];
  // fake_point points[N];

  scalar_t scalars[N];
  affine_t points[N];

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

  scalars[0] = {3827484040, 2874625294, 4134484179, 2522098891, 1684039639, 4190761864, 1674792009, 1733172596};
  scalars[1] = {2859778174, 247198543, 1069683537, 986951671, 18230349, 1865405355, 3834758898, 1605705230};
  scalars[2] = {870702623, 2140804196, 1118323047, 4097847923, 733285949, 2517599604, 2748585063, 1465198310};
  scalars[3] = {2997668077, 4130616472, 4255624276, 3713720096, 813455961, 1818410993, 1796699074, 1289452986};

  points[0].x = {927572523, 3123811016, 1178179586, 448270957, 3269025417, 873655910, 946685814, 846160237, 2311665546, 894701547, 1123227996, 414748152};
  points[0].y = {2100670837, 1657590303, 4206131811, 3111559769, 3261363570, 430821050, 2016803245, 2664358421, 3132350727, 189414955, 1844185218, 11036570};
  points[1].x = {606938594, 3862011666, 3396180143, 765820065, 3281167117, 634141057, 210831039, 670764991, 3442481388, 2417967610, 1382165347, 243748907};
  points[1].y = {2486871565, 3199940895, 3186416593, 2451721591, 4108712975, 2604984942, 1165376591, 854454192, 1479545654, 1006124383, 1570319433, 22366661};
  points[2].x = {183039612, 256454025, 4250922080, 2485970688, 3679755773, 1397028634, 1298805238, 3413182507, 2291846949, 1280816489, 1119750210, 122833203};
  points[2].y = {3025851512, 1147574033, 1323495323, 569405769, 382481561, 1330634004, 3879950484, 1158208050, 2740575984, 2745897444, 3101936482, 405605297};
  points[3].x = {4006417784, 3580973450, 2524244405, 3414509667, 4142213295, 3876406748, 4116037682, 877187559, 3606672288, 3459819278, 3198860768, 30571621};
  points[3].y = {182896763, 2741166359, 626891178, 1601768019, 1967793394, 706302600, 2612369182, 2051460370, 2918333441, 1902350841, 475238909, 239719017};

/*correct result:
1557917178, 269077943, 1116505460, 728110787, 4176849812, 3140203189, 2756051319, 197704154, 1838744007, 2201658078, 1505047534, 239949230, 
2029063365, 2557489072, 3905272471, 2418563649, 2077595491, 357415053, 3188715161, 1890916285, 354886608, 410171932, 1437862573, 206970588, 
4160033405, 2697065480, 1940009895, 2097886176, 4019146882, 2931880476, 3425684730, 2783686325, 1918054479, 1505257125, 3268347217, 269536830, */

  std::cout<<"scalars"<<std::endl;
  for (unsigned j = 0; j<N ; j++){

  for (unsigned i = 0; i < 8; i++) {
    std::cout << scalars[j].limbs_storage.limbs[i] << ", ";
  }
  std::cout << "\n";

  }

  std::cout<<"points"<<std::endl;
  for (unsigned j = 0; j<N ; j++){
  for (unsigned i = 0; i < 12; i++) {
    std::cout << points[j].x.limbs_storage.limbs[i] << ", ";
  }
  std::cout << "\n";
  for (unsigned i = 0; i < 12; i++) {
    std::cout << points[j].y.limbs_storage.limbs[i] << ", ";
  }
  std::cout << "\n";

  // return 0;
  // for (unsigned i = 0; i < 8; i++) {
  //   std::cout << points[j].z.limbs_storage.limbs[i] << ", ";
  // }
  // std::cout << "\n";
  }
  
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

  // scalars[0].val = 456;
  // scalars[1].val = 51;
  // scalars[2].val = 984;
  // scalars[3].val = 15;

  // points[0].val = 0;
  // points[1].val = 1;
  // points[2].val = 2;
  // points[3].val = 3;
  // cuda_ctx my_ctx = cuda_ctx(0);
  // large_msm<fake_point, fake_scalar>(my_ctx, points, scalars, N);

  // bucket_method_msm<scalar_t,projective_t, affine_t>(255, 10, scalars, points, N);
  // projective_t pr=short_msm<scalar_t,projective_t,affine_t>(scalars, points, N);

  projective_t *short_res = (projective_t*)malloc(sizeof(projective_t));
  projective_t *large_res = (projective_t*)malloc(sizeof(projective_t));

  // projective_t short_res[1];
  // projective_t large_res[1];

  short_msm<scalar_t, projective_t, affine_t>(scalars, points, N, short_res);
  large_msm<scalar_t, projective_t, affine_t>(scalars, points, N, large_res);

  std::cout<<"final result short"<<std::endl;
  std::cout<<*short_res<<std::endl;
  std::cout<<"final result large"<<std::endl;
  std::cout<<*large_res<<std::endl;

  // std::cout<<"final result short"<<std::endl;
  // std::cout<<pr<<std::endl;

  return 0;
}