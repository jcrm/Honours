#ifndef _INITIAL_FLOAT_
#define _INITIAL_FLOAT_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../Source/CUDA/cuda_header.h"


__global__ void cuda_kernel_initial_float(float *input, Size size, float value){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		//location is z slide + y position + variable size time x position
		float *cell_value = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
		cell_value[0] = value;
		cell_value[1] = 0.f;
	}
}
extern "C"
void cuda_fluid_initial_float(void *input, Size size, float value){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_initial_float<<<Dg,Db>>>((float*)input, size, value);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_initial() failed to launch error = %d\n", error);
	}
}
#endif