
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdio.h>
#define TPB 32
#define ATOMIC 1 // 0 for non-atomic addition

__global__
void dotKernel(float *d_result, float *d_array_a, float *d_array_b, int n) {
	__shared__ float s_product[TPB];
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;
	const int s_idx = threadIdx.x;
	
	if (idx >= n) {
		s_product[s_idx] = 0;
	}
	else {
	s_product[s_idx] = d_array_a[idx] * d_array_b[idx];
	}
	__syncthreads();


	// shared memory atomicAdd code 
	if (s_idx == 0) {
		float blockSum = 0.0;
		for (int j = 0; j < blockDim.x; ++j) {
			blockSum += s_product[j];
		}
		// Try each of two versions of adding to the accumulator
		if (ATOMIC) {
			atomicAdd(d_result, blockSum);
		}
		else {
			*d_result += blockSum;
		}
	}
}


void dotProduct(float *result, float *array_a, float *array_b, int n) {
	float *d_result;
	float *d_array_a;
	float *d_array_b;
	
	//create event variable for timing
	cudaEvent_t startKernel, stopKernel;
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);

	// Allocate memory for device arrays
	cudaMalloc(&d_result, sizeof(float));
	cudaMalloc(&d_array_a, n*sizeof(float));
	cudaMalloc(&d_array_b, n*sizeof(float));

	// Copy inout from host to device 
	cudaMemset(d_result, 0.0, sizeof(float));
	cudaMemcpy(d_array_a, array_a, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_array_b, array_b, n*sizeof(float), cudaMemcpyHostToDevice);
	
	// set shared memory size in byte 
	const size_t smemSize = TPB*sizeof(float);

	// Launch kernel to compute and store values 
	cudaEventRecord(startKernel);
	dotKernel << <(n+TPB-1)/TPB, TPB, smemSize >> >(d_result, d_array_a, d_array_b, n);
	cudaEventRecord(stopKernel);
	cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
	// Ensure times events have stoped 
	cudaEventSynchronize(stopKernel);

	// convert event records to time and output 
	float kernelTimeInMs = 0;
	cudaEventElapsedTime(&kernelTimeInMs, startKernel, stopKernel);
	printf("Kernel time with share memory(ms): %f\n\n", kernelTimeInMs);

	cudaFree(d_result);
	cudaFree(d_array_a);
	cudaFree(d_array_b);
}

__global__
void dotNoSKernel(float *d_result_NoS, float *d_array_a, float *d_array_b, int n) {
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= n) return;
	const int s_idx = threadIdx.x;

	__syncthreads();

	// atomicAdd without shared memory 
	atomicAdd(d_result_NoS, d_array_a[idx] * d_array_b[idx]);

}

void dotProductNoS(float *result_NoS, float *array_a, float *array_b, int n) {
	float *d_result_NoS;
	float *d_array_a;
	float *d_array_b;

	//create event variable for timing
	cudaEvent_t startKernelNoS, stopKernelNoS;
	cudaEventCreate(&startKernelNoS);
	cudaEventCreate(&stopKernelNoS);

	// Allocate memory for device arrays
	cudaMalloc(&d_result_NoS, sizeof(float));
	cudaMalloc(&d_array_a, n*sizeof(float));
	cudaMalloc(&d_array_b, n*sizeof(float));

	// Copy inout from host to device 
	cudaMemset(d_result_NoS, 0.0, sizeof(float));
	cudaMemcpy(d_array_a, array_a, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_array_b, array_b, n*sizeof(float), cudaMemcpyHostToDevice);

	// set shared memory size in byte 

	// Launch kernel to compute and store values 
	cudaEventRecord(startKernelNoS);
	dotNoSKernel << <(n + TPB - 1) / TPB, TPB >> >(d_result_NoS, d_array_a, d_array_b, n);
	cudaEventRecord(stopKernelNoS);
	cudaMemcpy(result_NoS, d_result_NoS, sizeof(float), cudaMemcpyDeviceToHost);
	// Ensure times events have stoped 
	cudaEventSynchronize(stopKernelNoS);

	// convert event records to time and output 
	float kernelTimeInMsNoS = 0;
	cudaEventElapsedTime(&kernelTimeInMsNoS, startKernelNoS, stopKernelNoS);
	printf("Kernel time without shared memory(ms): %f\n\n", kernelTimeInMsNoS);

	cudaFree(d_result_NoS);
	cudaFree(d_array_a);
	cudaFree(d_array_b);
}

