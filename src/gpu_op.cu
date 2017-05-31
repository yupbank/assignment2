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

// arr, value
// arr[:] = value
__global__ void array_set(float *output, const float input, int64_t n)
{
	int out_index = blockDim.x * blockIdx.x + threadIdx.x;
	if (out_index < n) {
		output[out_index] = input;
	}
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < arr->ndim; i++) {
    size *= arr->shape[i];
  }
  
  array_set<<<500, size/500>>>((float *)arr->data, value, size);
  return 0;
}

// input, output
// output[:,] = input 
__global__ void array_broadcast(float *output, const float *input, int row, int col, int new_dimension){
	int in_index = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i=0; i< new_dimension; i++){
		int out_index = i*row*col+in_index;
		output[out_index] = input[in_index];
	}

}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2);
  assert(output->ndim == 3);
  assert(input->shape[0] == output->shape[1] &&
         input->shape[1] == output->shape[2]);

  int nrow = input->shape[0];
  int ncol = input->shape[1];
  int new_dimension = output->shape[0];
  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;

  array_broadcast<<<nrow, ncol>>>(output_data, input_data, nrow, ncol, new_dimension);

  return 0;
}

__global__ void value_add_keneral(float *input, float *output, int64_t size, int64_t rows)
{
	int stride = blockDim.x;
	int id = threadIdx.x;
	for(int i= id; i<size; i+=stride){
		float v = 0.0;
		for (int j=0; j<rows; j++)
		{
			v += input[j*size+i];
		}
		output[i]=v;
	}
}

// output = input.sum(axis=0)

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size =input->shape[1] *  input->shape[2];
  value_add_keneral<<<1, 1024>>>((float *)input->data, (float *)output->data, size, output->shape[0]);

  return 0;
}

__global__ void matAdd(float * A, float *B, float *C, int64_t size) 
{ 
	int begin = threadIdx.x; 
	int stride = blockDim.x;
	for (int i=begin; i<size; i+=stride){
		C[i] = A[i]+B[i];
	}
}


// output = matA+matB
int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
// dim3 thread(matA.shape[0], matA.shape[1]);
  int64_t size = 1;
  for (int i = 0; i < matA->ndim; i++) {
    size *= matA->shape[i];
  }
  matAdd<<<1, 1024>>>((float *) matA->data, (float *) matB->data, (float *) output->data, size);
  return 0;
}
__global__ void elementAdd(float *input, float value, float *output, int64_t size) 
{ 
	int i = threadIdx.x; 
	int stride = blockDim.x;
	for (int b=i; b<size; b+= stride){
		output[b] = input[b]+value;
	}
}
int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) 
{ /* TODO: Your code here */

  int64_t size = 1;
  for (int i = 0; i < input->ndim; i++) {
    size *= input->shape[i];
  }
	elementAdd<<<1, 1024>>>((float *)input->data, val, (float *)output->data, size);
	return 0;
}

__global__ void MatMultiply(float * A, float *B, float *C, int64_t size) 
{ 
	int begin = threadIdx.x; 
	int stride = blockDim.x;
	for (int i=begin; i<size; i+=stride)
	{
		C[i] = A[i]*B[i];
	}
}
int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < matA->ndim; i++) {
    size *= matA->shape[i];
  }
  MatMultiply<<<1, 1024>>>((float *)matA->data, (float *)matB->data, (float *)output->data, size);
  return 0;
}

__global__ void elementMultiply(float *input, float value, float *output, int64_t size) 
{ 
	int i = threadIdx.x; 
	int stride = blockDim.x;
	for (int b=i; b<size; b+= stride){
		output[b] = input[b]*value;
	}
}
int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < input->ndim; i++) {
    size *= input->shape[i];
  }
	elementMultiply<<<1, 1024>>>((float *)input->data, val, (float *)output->data, size);

  return 0;
}


int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
	int m = matA->shape[0];
	int n = matA->shape[1];
	int k = matB->shape[1];
	int lda = 0, ldb=0, ldc=0;
	float alpha = 1;
	float beta = 0;
  cublasHandle_t handle;
  cublasCreate(&handle);
  if (transposeA)
	{
	  lda = m;
		if (transposeB) {
			ldb = k;
			ldc = k;
			  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, (float *) matA->data, lda, (float *) matB->data, ldb, &beta, (float *) matC->data, ldc); 
		
		}
		else {
			ldb = k;
			ldc = k;
			  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, (float *) matA->data, lda, (float *) matB->data, ldb, &beta, (float *) matC->data, ldc); 
		}
	}
	else {
	  lda = m;
		if (transposeB) {
			ldb = k;
			ldc = k;
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, (float *) matA->data, lda, (float *) matB->data, ldb, &beta, (float *) matC->data, ldc); 
		}
		else {
			ldb = k;
			ldc = k;
			  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, (float *) matA->data, lda, (float *) matB->data, ldb, &beta, (float *) matC->data, ldc); 
		}
	
	}
  return 0;
}
__global__ void relu(float *input, float *output, int64_t size) {
	int i = threadIdx.x; 
	int stride = blockDim.x;
	for (int b=i; b<size; b+= stride){
		output[b] = max(input[b], 0.0f);
	}
	
}
// ln(1+e^x)
int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */

  int64_t size = 1;
  for (int i = 0; i < input->ndim; i++) {
    size *= input->shape[i];
  }
	relu<<<1, 1024>>>((float *)input->data, (float *)output->data, size);
	return 0;
}

__global__ void relu_gradient(float *input, float *in_grad, float *output, int64_t size){
	int i = threadIdx.x; 
	int stride = blockDim.x;
	for (int b=i; b<size; b+= stride){
		output[b] = input[b] > 0 ? in_grad[b]: 0.0;
	}

}
// 1/(1+e^(-x))
int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < input->ndim; i++) {
    size *= input->shape[i];
  }

	relu_gradient<<<1, 1024>>>((float *)input->data, (float *) in_grad->data, (float *)output->data, size);
	return 0;
}

__global__ void softmax(float *input, float *output, int nrow, int ncol) {
	int i = threadIdx.x; 
	int stride = blockDim.x;
	for (int b=i; b<nrow; b+= stride){
	float sum = 0;
	for (int m=0; m<ncol; m++)
	{
		sum += exp(input[b*ncol+m]);
	}	
	for (int m=0; m<ncol; m++) {
		output[b*ncol+m] = exp(input[b*ncol+m])/sum;
	}
	}

}

// e^x[0]/sum(e^x[i])
int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t nrow = input->shape[0];
  int64_t ncol = input->shape[1];
   softmax<<<1, 1024>>>((float *) input->data, (float *) output->data, nrow, ncol);
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
