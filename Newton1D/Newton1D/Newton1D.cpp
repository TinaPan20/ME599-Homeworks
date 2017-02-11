// Kaicen Pan  Main cpp file 
# include "calculation.h"
# include <stdlib.h>
# include <iostream>
# define N 100

void print(float *array, int len){
	for (int i = 0; i < len; i++)
	{
		printf(" %f ", array[i]);
	}
	printf("\n\n");
}

int main()
{

	float *in1 = (float*)calloc(N, sizeof(float));
	float *out = (float*)calloc(N, sizeof(float));

	// Compute scaled input values
	in1[0] = -5.0;
	for (int i = 1; i <= N; ++i)
	{
		in1[i] = in1[i - 1] + 0.1;
	}

	newtonArray(out, in1, N);

	system("PAUSE");

	free(in1);
	free(out);
	return 0;
}