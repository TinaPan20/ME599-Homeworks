#include "kernel.h"
#define TX 32
#define TY 32
#define LEN 5.f
#define TIME_STEP 0.05f
#define FINAL_TIME 10.f
#include <stdio.h>
#include <math.h>

// scale coordinates onto [-LEN, LEN]
__device__
float scale(int i, int w) { return 2 * LEN*(((1.f*i)/w) - 0.5f); }

// function for right-hand side of y-equation
__device__
float f(float x, float y, float param, float sys) {
  if (sys == 1) return x - 2 * param*y; // negative stiffness
  if (sys == 2) return -x + param*(1 - x*x)*y; //van der Pol
  else return -x - 2 * param*y;
}

__device__
float g(float y){
	return y;
}


// explicit Runge Kutta solver
__device__
float2 RK(float x, float y, float dt, float tFinal, float param, float sys) {
	float dx = 0.f, dy = 0.f;
	float k1; float k2; float k3; float k0; float l0; float l1; float l2; float l3;
	for (float t = 0; t < tFinal; t += dt) {
		k0 = g(y);
		l0 = f(x, y, param, sys);

		k1 = g(y + 0.5*l0*dt);
		l1 = f(x + 0.5*k0*dt, y + 0.5*l0*dt, param, sys);

		k2 = g(y + 0.5*l1*dt);
		l2 = f(x + 0.5*k1*dt, y + 0.5*l1*dt, param, sys);

		k3 = g(y + l2*dt);
		l2 = f(x + k2*dt, y + l2*dt, param, sys);

		dx = dt * 1 / 6 * (k0 + 2 * k1 + 2 * k2 + k3);
		dy = dt * 1 / 6 * (l0 + 2 * l1 + 2 * l2 + l3);
		x += dx;
		y += dy;
	}
	return make_float2(x, y);
}


// explicit Euler solver
__device__
float2 euler(float x, float y, float dt, float tFinal, float param, float sys) {
	float dx = 0.f, dy = 0.f;
	for (float t = 0; t < tFinal; t += dt) {
		dx = dt*y;
		dy = dt*f(x, y, param, sys);
		x += dx;
		y += dy;
	}
	return make_float2(x, y);
}

__device__
unsigned char clip(float x){ return x > 255 ? 255 : (x < 0 ? 0 : x); }

// kernel function to compute decay and shading
__global__
void stabImageKernel(uchar4 *d_out, int w, int h, float p, int s) {
  const int c = blockIdx.x*blockDim.x + threadIdx.x;
  const int r = blockIdx.y*blockDim.y + threadIdx.y;
  if ((c >= w) || (r >= h)) return; // Check if within image bounds
  const int i = c + r*w; // 1D indexing
  const float x0 = scale(c, w);
  const float y0 = scale(r, h);
  const float dist_0 = sqrt(x0*x0 + y0*y0);
  const float2 pos = RK(x0, y0, TIME_STEP, FINAL_TIME, p, s);
  const float dist_f = sqrt(pos.x*pos.x + pos.y*pos.y);
  // assign colors based on distance from origin
  const float dist_r = dist_f / dist_0;

  d_out[i].x = clip(dist_r * 255); // red ~ growth
  d_out[i].y = ((c == w / 2) || (r == h / 2)) ? 255 : 0; // axes
  d_out[i].z = clip((1 / dist_r) * 255); // blue ~ 1/growth
  d_out[i].w = 255;
  if (c == w*0.55 && r == h*0.55){
	  printf("distance= %f", dist_f);
  }
   
}

void kernelLauncher(uchar4 *d_out, int w, int h, float p, int s) {
  
	cudaEvent_t startKernel, stopKernel;
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);

	const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3((w + TX - 1)/TX, (h + TY - 1)/TY);
  stabImageKernel<<<gridSize, blockSize >>>(d_out, w, h, p, s);
  cudaEventSynchronize(stopKernel);
}