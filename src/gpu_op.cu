#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 512

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}


__global__ void array_set_kernel(float *data, float value, int64_t size) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < size; i += stride) {
    data[i] = value;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < arr->ndim; i++) {
    size *= arr->shape[i];
  }
  array_set<<<1, 1024>>>(value, (float *)arr->data, size);
  return 0;
}
// arr, value
// arr[:] = value
__global__ void array_set(const float input, float *output, int64_t n)
{
	int out_index = blockDim.x * blockIdx.x + threadIdx.x;
	if (out_index < n) {
		output[out_index] = input;
	}
}
//int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
//  printf("value : %f ", value);
//
//  int size = 1;
//  for (int i=0; i<arr->ndim; i++) 
//  {
//	  size *= int(arr->shape[i]);
//  }
//  arr->data = (float*)malloc(size*sizeof(float));
//  for (int i=0; i<=size; i++)
//  {
//	  arr->data[i] = value;
//  }
//  //printf("value : %d ", size);
//  //array_set<<<1, size>>>(value, output_data, size);
//  return 0;
//}

// input, output
// output[:,] = input 
int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 3);
  assert(input->shape[0] == output->shape[1] &&
         input->shape[1] == output->shape[2]);

  //int nrow = input->shape[0];
  //int ncol = input->shape[1];
  //int new_dimension = output->shape[0];
  //float *input_data = (float *)input->data;
  //float *output_data = (float *)output->data;

  //dim3 threads;
  //threads.x = nrow;
  //threads.y = ncol;
  //float* d_input;
  //int size = nrow*ncol;
  //cudaMalloc(&d_input, size);
  //cudaMemcpy(d_input, input_data, size, cudaMemcpyHostToDevice);
  //// every new_dimension there is a block;
  //// every block there is (x, y) threads
  //// every thread only copy one cell
  //value_copy_kernel<<<new_dimension, threads>>>(
  //    nrow, ncol, d_input, output_data);

  return 0;
}

//__global__ void value_add_keneral(float *input, float *output)
//{
//  extern __shared__ float sum_per_dim = 0.0;
//  sum_per_dim += input[threadIdx.x][blockIdx.x][blockIdx.y]; 
//  __syncthreads();
//  output[blockIdx.x][blockIdx.y] = sum_per_dim
//}

// output = input.sum(axis=0)

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */

  //value_add_keneral<<<(input->shape[1], input->shape[2]), input->shape[0]>>>(input, output)

  return 0;
}

//__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) 
//{ 
//	int i = threadIdx.x; 
//	int j = threadIdx.y;
//       	C[i][j] = A[i][j] + B[i][j];
//}


// output = matA+matB
int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
// dim3 thread(matA.shape[0], matA.shape[1]);
//  MatAdd<<<1, thread>>>(matA, matB, output);
  return 0;
}
//__global__ void elementAdd(float *input, float value, float *output) 
//{ 
//	int i = threadIdx.x; 
//	int j = threadIdx.y;
//       	output[i][j] = input[i][j] + value;
//
int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) 
{ /* TODO: Your code here */
	//dim3 thread(input->shape[0], input->shape[1])
	//	elementAdd<<<1, thread>>>(matA, matB, output);
	return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */

  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  return 0;
}

// ln(1+e^x)
int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

// 1/(1+e^(-x))
int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

// e^x[0]/sum(e^x[i])
int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
