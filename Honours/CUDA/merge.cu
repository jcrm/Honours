#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "math.h"
#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_merge(float *output, Size size){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;

	float* rain = output + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
	float sum = rain[0] + rain[1] + rain[2] + rain[3];
	rain[0] = sum;
	rain[1] = 0.f;
	rain[2] = 0.f;
	rain[3] = 0.f;
}
extern "C"
void cuda_fluid_merge(void *output, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	//dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_merge<<<Dg,Db>>>((float *)output, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}