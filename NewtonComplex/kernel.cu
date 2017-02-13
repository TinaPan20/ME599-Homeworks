#include "kernel.h"
#include <stdio.h>
#define TX 32
#define TY 32
#define LEN 5.f
#define error 0.001f
#define iterationTime 40


__device__
float scale(int i, int w) { return 2 * LEN*(((1.f*i) / w) - 0.5f); }

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
int2 newtonComplex(const float2 z){
	// define 3 correct roots and their values 
	float2 r1, r2, r3;
	r1.x = 1.f; r1.y = 0.f;
	r2.x = -0.5f; r2.y = 0.866f;
	r3.x = -0.5f; r3.y = -0.866f;

	float x = z.x;
	float y = z.y;
	float u, ux, uy, v, vx, vy, x1, y1;
	float ratio_x, ratio_y;
	float dist1;
	float dist2;
	float dist3;
	int a;
	int colorcode; 

	for (int i = 0; i < iterationTime; i++){
		a = i;
		u = x*x*x - 3 * x*y*y - 1;
		ux = 3 * x*x - 3 * y*y;
		uy = -6 * x*y;
		v = 3 * x*x*y - y*y*y;
		vx = 6 * x*y;
		vy = 3 * x*x - 3 * y*y;
		ratio_x = (u*ux + v*vx) / (ux*ux + vx*vx);
		ratio_y = (u*uy + v*vy) / (uy*uy + vy*vy);
		x = x - ratio_x;
		y = y - ratio_y;
		dist1 = sqrt((x - r1.x)*(x - r1.x) + (y - r1.y)*(y - r1.y));
		dist2 = sqrt((x - r2.x)*(x - r2.x) + (y - r2.y)*(y - r2.y));
		dist3 = sqrt((x - r3.x)*(x - r3.x) + (y - r3.y)*(y - r3.y));
		i++;
		if (dist1 < error || dist2 < error || dist3 < error){
			if (dist1 < error){ colorcode = 1; }
			if (dist2 < error){ colorcode = 2; }
			if (dist3 < error){ colorcode = 3; }
			break;
		}
	}

	int2 result;
	result.x = colorcode;
	result.y = a;
	return result;
}


__global__
void distanceKernel(uchar4 *d_out, int w, int h, int2 pos) {
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	if ((c >= w) || (r >= h)) return; // Check if within image bounds
	const int i = c + r*w; // 1D indexing
	
	float2 z0;
	const float x0 = scale(c - pos.x + w / 2, w);
	const float y0 = scale(r - pos.y + h / 2, h);
	z0.x = x0;
	z0.y = y0;
	int color;
	int2 z = newtonComplex(z0);
	color = z.x;

	float intensity = (1.f*iterationTime - z.y) / iterationTime;
	if (color == 1){
		d_out[i].x = intensity * 255; 
		d_out[i].y = 0;
		d_out[i].z = 0; 
		d_out[i].w = 255;
	}
	if (color == 2){
		d_out[i].x = 0;
		d_out[i].y = intensity * 255; 
		d_out[i].z = 0; 
		d_out[i].w = 255;
	}
	if (color == 3){
		d_out[i].x = 0; 
		d_out[i].y = 0; 
		d_out[i].z = intensity * 255; 
		d_out[i].w = 255;
	}
}

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos) {
	const dim3 blockSize(TX, TY);
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);
	distanceKernel << <gridSize, blockSize >> >(d_out, w, h, pos);
}



