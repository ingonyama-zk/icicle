

#define SHMEM_SIZE 64
#define MAX_SHMEM_LOG_SIZE 6


template <typename S>
__global__ void mult_and_reduce(S *v, S *v_r, S alpha, int nof_results) {
	// Allocate shared memory
	__shared__ S partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	// partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	// partial_sum[threadIdx.x] = (S::one() - alpha) * v[2*i] + alpha * v[2*i+1] + (S::one() - alpha) * v[2*(i + blockDim.x)] + alpha * v[2*(i + blockDim.x)+1];
	S e1 = v[2*i] + alpha * (v[2*i+1] - v[2*i]);
	S e2 = v[2*(i + blockDim.x)] + alpha * (v[2*(i + blockDim.x)+1] - v[2*(i + blockDim.x)]);
	// S e1 = v[2*i] + (v[2*i+1] - v[2*i]);
	// S e2 = v[2*(i + blockDim.x)] + (v[2*(i + blockDim.x)+1] - v[2*(i + blockDim.x)]);
	// partial_sum[threadIdx.x] = v[2*i] + v[2*(i + blockDim.x)] + alpha * (v[2*i+1] - v[2*i] + v[2*(i + blockDim.x)+1] - v[2*(i + blockDim.x)]);
	partial_sum[threadIdx.x] = e1 + e2;
	v[i] = e1;
	v[i + blockDim.x] = e2;
	// for (int j = 0; j < 2; j++)
	// {
	// 	partial_sum[threadIdx.x] = partial_sum[threadIdx.x] * partial_sum[threadIdx.x];
	// }
	
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 1; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x < nof_results) {
		// printf("debug tid %d, val %d\n", threadIdx.x, partial_sum[threadIdx.x]);
		v_r[nof_results*blockIdx.x + threadIdx.x] = partial_sum[threadIdx.x];
	}
}


template <typename S>
__global__ void sum_reduction(S *v, S *v_r, int nof_results) {
	// Allocate shared memory
	__shared__ S partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Load elements AND do first add of reduction
	// Vector now 2x as long as number of threads, so scale i
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[i] + v[i + blockDim.x];
	__syncthreads();

	// Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 1; s >>= 1) {
		// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] = partial_sum[threadIdx.x] + partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x < nof_results) {
		// printf("debug tid %d, val %d\n", threadIdx.x, partial_sum[threadIdx.x]);
		v_r[nof_results*blockIdx.x + threadIdx.x] = partial_sum[threadIdx.x];
	}
}

template <typename S>
__global__ void update_evals_kernel(S* evals, S alpha){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // evals[tid] = (S::one() - alpha) * evals[2*tid] + alpha * evals[2*tid+1];
  // evals[tid] =  evals[2*tid] + (evals[2*tid+1] - evals[2*tid]);
  evals[tid] =  evals[2*tid] + alpha * (evals[2*tid+1] - evals[2*tid]);
  // evals[tid] = (1 - alpha) * evals[2*tid] + alpha * evals[2*tid+1];
}

template <typename S>
void accumulate(S* in, S* out, uint32_t log_size, int nof_results, cudaStream_t stream){
  uint32_t nof_steps = (log_size - 1) / MAX_SHMEM_LOG_SIZE;
  uint32_t last_step_size = (log_size - 1) % MAX_SHMEM_LOG_SIZE;
	printf("a nof steps %d last size %d\n", nof_steps, last_step_size);
  for (int i = 0; i < nof_steps; i++)
  {
    sum_reduction<<<1<<(log_size - 1 -(MAX_SHMEM_LOG_SIZE)*(i+1)), SHMEM_SIZE,0,stream>>>(i? out : in, out, nof_results);
		printf("a nof blocks %d\n", 1<<(log_size - 1 -(MAX_SHMEM_LOG_SIZE)*(i+1)));
		// cudaDeviceSynchronize();
  	// printf("cuda err %d\n", cudaGetLastError());
  }
  if (last_step_size) sum_reduction<<<1, 1<<last_step_size, 0,stream>>>(nof_steps? out : in, out, nof_results);
	// cudaDeviceSynchronize();
  // printf("cuda err last %d\n", cudaGetLastError());
}

template <typename S>
void mult_and_accumulate(S* in, S* out, uint32_t log_size, S alpha, int nof_results, cudaStream_t stream){
  uint32_t nof_steps = (log_size - 1) / MAX_SHMEM_LOG_SIZE;
  uint32_t last_step_size = (log_size - 1) % MAX_SHMEM_LOG_SIZE;
	printf("m nof steps %d last size %d\n", nof_steps, last_step_size);
  for (int i = 0; i < nof_steps; i++)
  {
		if (i) sum_reduction<<<1<<(log_size - 1 -(MAX_SHMEM_LOG_SIZE)*(i+1)), SHMEM_SIZE,0,stream>>>(i? out : in, out, nof_results);
    else mult_and_reduce<<<1<<(log_size - 1 -(MAX_SHMEM_LOG_SIZE)*(i+1)), SHMEM_SIZE,0,stream>>>(i? out : in, out, alpha, nof_results);
		printf("m nof blocks %d\n", 1<<(log_size-(MAX_SHMEM_LOG_SIZE)*(i+1)-1));
		// cudaDeviceSynchronize();
  	// printf("cuda err %d\n", cudaGetLastError());
  }
  if (last_step_size) {
		if (nof_steps) sum_reduction<<<1, 1<<last_step_size, 0,stream>>>(nof_steps? out : in, out, nof_results);
		else mult_and_reduce<<<1, 1<<last_step_size, 0,stream>>>(nof_steps? out : in, out, alpha, nof_results);
	} 
	// cudaDeviceSynchronize();
  // printf("cuda err last %d\n", cudaGetLastError());
}

template <typename S>
__global__ void add_to_trace(S* trace, S* vals, int p){
	  trace[2*p+1] = vals[0];
    trace[2*p+2] = vals[1];
		// printf("%d  %d\n", vals[0], vals[1]);
}

template <typename S>
// __global__ void combinations_kernel(S* in, S* out, S (*combine_func)()){
__global__ void combinations_kernel3(S* in, S* out, int poly_size){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	S rp[3] = {S::one, S::one, S::one, S::one};
	S e1, e2;
	#pragma unroll
	for (int l = 0; l < 3; l++)
	{
		rp[0] *= in[l*poly_size + 2*tid];
	}
	#pragma unroll
	for (int l = 0; l < 3; l++)
	{
		rp[1] *= in[l*poly_size + 2*tid + 1];
	}
	#pragma unroll
	for (int l = 0; l < 3; l++)
	{
		e1 = in[l*poly_size + 2*tid];
		e2 = in[l*poly_size + 2*tid + 1];
		rp[2] *= e2 + e2 - e1;
	}
	#pragma unroll
	for (int l = 0; l < 3; l++)
	{
		e1 = in[l*poly_size + 2*tid];
		e2 = in[l*poly_size + 2*tid + 1];
		rp[3] *= e2 + e2 + e2 - e1 - e1;
	}
	out[4*tid] = rp[0];
	out[4*tid+1] = rp[1];
	out[4*tid+2] = rp[2];
	out[4*tid+3] = rp[3];
}

// template <typename S, int M>
// // __global__ void combinations_kernel(S* in, S* out, S (*combine_func)()){
// __global__ void combinations_kernel(S* in, S* out){
// 	int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 	S rp = s::one;
// 	#pragma unroll
// 	for (int k = 0; k < M+1; k++)
// 	{
// 		#pragma unroll
// 		for (int l = 0; l < M; i++)
// 		{
// 			rp *= in[2*tid] * (1 - k) + in[2*tid + 1] * k;
// 		}
// 	}
// }

// template <typename S>
// __device__ S simple_combine(S* f){
// 	return f[0]*f[1]*f[2]
// }

template <typename S>
S my_hash(){
	return 1;
}

template <typename S>
void sumcheck_alg1(S* evals, S* t, S* T, S C, int n, cudaStream_t stream){
	// S alpha = 1;
	S alpha = S::one();
	// S alpha = S::rand_host();
  // S alpha = my_hash(/*T, C*/);
  // S rp_even, rp_odd;
  for (int p = 0; p < n-1; p++)
  {
    int nof_threads = 1<<(n-1-p);
    // move update kernel here and unify
    // reduction_kernel<<<nof_threads>>>(evals, t, n-p); //accumulation
    accumulate(evals, t, n-p, 2, stream); //accumulation
		add_to_trace<<<1,1,0,stream>>>(T, t, p);
    // T[2*p+1] = t[0];
    // T[2*p+2] = t[1];
    // alpha = my_hash(/*alpha, t[0], t[1]*/); //phase 2
		int NOF_THREADS = 256;
		int NOF_BLOCKS = (nof_threads + NOF_THREADS - 1) / NOF_THREADS;
    update_evals_kernel<<<NOF_BLOCKS, NOF_THREADS,0, stream>>>(evals, alpha); //phase 3
  }
	add_to_trace<<<1,1,0,stream>>>(T, evals, n-1);

}

template <typename S>
void sumcheck_alg1_unified(S* evals, S* t, S* T, S C, int n, cudaStream_t stream){
	// S alpha = 1;
	S alpha = S::one();
	// S alpha = S::rand_host();
  // S alpha = my_hash(/*T, C*/);
  // S rp_even, rp_odd;
  for (int p = 0; p < n-1; p++)
  {
    int nof_threads = 1<<(n-1-p);
		printf("nof threads %d\n", nof_threads);
    // move update kernel here and unify
    // reduction_kernel<<<nof_threads>>>(evals, t, n-p); //accumulation
    if (p) mult_and_accumulate(evals, t, n-p, alpha, 2, stream); //accumulation
		else accumulate(evals, t, n-p, 2, stream);
		add_to_trace<<<1,1,0,stream>>>(T, t, p);
    // T[2*p+1] = t[0];
    // T[2*p+2] = t[1];
    // alpha = my_hash(/*alpha, t[0], t[1]*/); //phase 2
		// int NOF_THREADS = 256;
		// int NOF_BLOCKS = (nof_threads + NOF_THREADS - 1) / NOF_THREADS;
    // update_evals_kernel<<<NOF_BLOCKS, NOF_THREADS,0, stream>>>(evals, alpha); //phase 3
  }
	update_evals_kernel<<<1, 2,0, stream>>>(evals, alpha);
	add_to_trace<<<1,1,0,stream>>>(T, evals, n-1);
}

// template <typename S>
// void sumcheck_alg3_unified(S* evals, S* t, S* T, S C, int n, cudaStream_t stream){
// 	// S alpha = 1;
// 	S alpha = S::one();
// 	// S alpha = S::rand_host();
//   // S alpha = my_hash(/*T, C*/);
//   // S rp_even, rp_odd;
//   for (int p = 0; p < n-1; p++)
//   {
//     int nof_threads = 1<<(n-1-p);
// 		printf("nof threads %d\n", nof_threads);
//     // move update kernel here and unify
//     // reduction_kernel<<<nof_threads>>>(evals, t, n-p); //accumulation
//     if (p) mult_and_accumulate(evals, t, n-p, alpha, stream); //accumulation
// 		else accumulate(evals, t, n-p, stream);
// 		add_to_trace<<<1,1,0,stream>>>(T, t, p);
//     // T[2*p+1] = t[0];
//     // T[2*p+2] = t[1];
//     // alpha = my_hash(/*alpha, t[0], t[1]*/); //phase 2
// 		// int NOF_THREADS = 256;
// 		// int NOF_BLOCKS = (nof_threads + NOF_THREADS - 1) / NOF_THREADS;
//     // update_evals_kernel<<<NOF_BLOCKS, NOF_THREADS,0, stream>>>(evals, alpha); //phase 3
//   }
// 	update_evals_kernel<<<1, 2,0, stream>>>(evals, alpha);
// 	add_to_trace<<<1,1,0,stream>>>(T, evals, n-1);
// }


template <typename S>
void sumcheck_alg1_ref(S* evals, S* t, S* T, S C, int n){
  // S alpha = my_hash(/*T, C*/);
	// S alpha = 1;
	S alpha = S::one();
  S rp_even, rp_odd;
  for (int p = 0; p < n; p++)
  {
		// rp_even = 0; rp_odd = 0;
		rp_even = S::zero(); rp_odd = S::zero();
		// printf("evals\n");
		// for (int i = 0; i < 1<<(n-p); i++)
		// {
		// 	printf("%d, ",evals[i]);
		// }
		// printf("\n");
		for (int i = 0; i < 1<<(n-1-p); i++)
		{
			rp_even = rp_even + evals[2*i];
			rp_odd = rp_odd + evals[2*i+1];
		}
    T[2*p+1] = rp_even;
    T[2*p+2] = rp_odd;
    // alpha = my_hash(/*alpha, t[0], t[1]*/); //phase 2
		// alpha = 1;
		alpha = S::one();
		for (int i = 0; i < 1<<(n-1-p); i++)
		{
			t[i] = (S::one() - alpha) * evals[2*i] + alpha * evals[2*i+1];
			// t[i] = (1-alpha)*evals[2*i] + alpha*evals[2*i+1];
		}
		for (int i = 0; i < 1<<(n-1-p); i++)
		{
			evals[i] = t[i];
		}
  }
}