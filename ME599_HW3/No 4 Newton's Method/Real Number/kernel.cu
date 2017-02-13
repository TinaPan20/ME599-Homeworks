#include "calculation.h"
#include <stdio.h>
#define N 32
#define TPB 32



__device__
float newton(float x1){
	float x3, x2;
	x2 = x1 - (x1*x1*x1 - x1) / (3 * x1*x1 - 1);
	x3 = x2 - (x2*x2*x2 - x2) / (3 * x2*x2 - 1);
	return x3;
}

__global__
void newtonKernel(float *d_out, float *d_in1)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i > 100) return;
	const float x = d_in1[i];
	d_out[i] = newton(x);
	if (fabs(d_out[i] - d_in1[i]) < 0.0001){
		printf("the real root is %f\n", d_out[i]);
	}
}

void newtonArray(float *out, float *in1, int len)
{
	// Declare pointers to device arrays
	float *d_in1;
	float *d_out;

	// Allocate memory for device arrays
	cudaMalloc(&d_in1, len*sizeof(float));
	cudaMalloc(&d_out, len*sizeof(float));

	// Copy input data from host to device
	cudaMemcpy(d_in1, in1, len*sizeof(float), cudaMemcpyHostToDevice);

	// Launch kernel to compute and store distance values
	newtonKernel << <(len + TPB - 1) / TPB, TPB >> >(d_out, d_in1);// , d_in2);

	// Copy results from device to host
	cudaMemcpy(out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost);

	// Free the memory allocated for device arrays
	cudaFree(d_in1);
	cudaFree(d_out);
}


