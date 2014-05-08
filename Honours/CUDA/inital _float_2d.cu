#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "../Source/CUDA/cuda_header.h"
#include "math.h"


__global__ void cuda_kernel_initial_float_2d(float *input, Size size, float value){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;

	//location is z slide + y position + variable size time x position
	float* rain = input + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
	rain[0] = value;
	rain[1] = value;
	rain[2] = value;
	rain[3] = value;
}
extern "C"
void cuda_fluid_initial_float_2d(void *input, Size size, float value){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_initial_float_2d<<<Dg,Db>>>((float*)input, size, value);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_initial() failed to launch error = %d\n", error);
	}
}