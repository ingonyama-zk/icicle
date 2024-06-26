
template <class T>
__global__ void add_elements_kernel(const T* x, const T* y, T* result, const unsigned count)
{
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  // printf("tid %d\n", tid);
  result[tid] = x[tid] + y[tid];
}

template <class T>
__global__ void fake_ntt_kernel(const T* x, T* result, const unsigned thread_count)
{
  extern __shared__ T shmem[];
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= thread_count) return;
  shmem[4*threadIdx.x] = x[4*tid] + x[4*tid+1];
  shmem[4*threadIdx.x+1] = x[4*tid] + T::neg(x[4*tid+1]);
  shmem[4*threadIdx.x+2] = x[4*tid+2] + x[4*tid+3];
  shmem[4*threadIdx.x+3] = x[4*tid+2] + T::neg(x[4*tid+3]);
  __syncthreads();
  result[4*tid] = shmem[2*threadIdx.x] + shmem[2*threadIdx.x + 4*blockDim.x/2];
  result[4*tid+1] = shmem[2*threadIdx.x] + T::neg(shmem[2*threadIdx.x + 4*blockDim.x/2]);
  result[4*tid+2] = shmem[2*threadIdx.x+1] + shmem[2*threadIdx.x + 4*blockDim.x/2+1];
  result[4*tid+3] = shmem[2*threadIdx.x+1] + T::neg(shmem[2*threadIdx.x + 4*blockDim.x/2+1]);
}

template <class T>
__global__ void bugged_fake_ntt_kernel(const T* x, T* result, const unsigned thread_count)
{
  extern __shared__ T shmem[];
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  // if (tid >= thread_count) return;
  // if (tid == 0){
  //   for (int i = 0; i < 8; i++)
  //   {
  //     shmem[i]=T::zero();
  //   }
  // }

  shmem[4*threadIdx.x] = x[4*tid] + x[4*tid+1];
  shmem[4*threadIdx.x+1] = x[4*tid] + T::neg(x[4*tid+1]);
  shmem[4*threadIdx.x+2] = x[4*tid+2] + x[4*tid+1];
  shmem[4*threadIdx.x+4] = x[4*tid+2] + T::neg(x[4*tid+1]);

  __syncthreads();

  // if (tid == 0){
  //   for (int i = 0; i < 8; i++)
  //   {
  //     printf("%d ",shmem[i]);
  //   }
  //   printf("\n");
  // }

  // printf("tid: %d, addr1: %d, addr2: %d\n", tid, 2*threadIdx.x, 2*threadIdx.x + 4*blockDim.x);
  result[4*tid] = shmem[2*threadIdx.x] + shmem[2*threadIdx.x + 4*blockDim.x];  // Incorrect offset
  result[4*tid+1] = shmem[2*threadIdx.x] + T::neg(shmem[2*threadIdx.x + 4*blockDim.x]);  // Incorrect offset
  result[4*tid+2] = shmem[2*threadIdx.x+1] + shmem[2*threadIdx.x + 4*blockDim.x+1];  // Incorrect offset
  result[4*tid+3] = shmem[2*threadIdx.x+1] + T::neg(shmem[2*threadIdx.x +4*blockDim.x+1]);  // Incorrect offset
}