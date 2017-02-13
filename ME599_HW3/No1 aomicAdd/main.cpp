#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#define N  10000


int main() {
	float cpu_result = 0.0;
	float gpu_result = 0.0;
	float gpu_result_NoS = 0.0;
	float *array_a = (float*)calloc(N, sizeof(float));
	float *array_b = (float*)calloc(N, sizeof(float));

	//Initialize input arrays
	for (int i = 0; i < N; ++i) {
		array_a[i] = 0.25;
		array_b[i] = 0.75;
	}
	clock_t CPUdotBegin = clock();
	for (int j = 0; j < N; ++j) {
		cpu_result += array_a[j] * array_b[j];
	}
	clock_t CPUdotEnd = clock();
	float CPUdotTime = (float(CPUdotEnd - CPUdotBegin)) / CLOCKS_PER_SEC;

	printf("cpu result = %f\n\n", cpu_result);
	printf("CPU time (ms): %f\n\n", CPUdotTime * 1000);

	dotProduct(&gpu_result, array_a, array_b, N);
	dotProductNoS(&gpu_result_NoS, array_a, array_b, N);

	printf("gpu result with shared memory = %f\n\n", gpu_result);
	printf("gpu result without shared memory = %f\n\n", gpu_result_NoS);

	system("pause");

	free(array_a);
	free(array_b);
	return 0;
}