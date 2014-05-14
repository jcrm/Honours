#ifndef _BOUNDARIES_THERMO_CUDA_
#define _BOUNDARIES_THERMO_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math.h"

#include "../Source/CUDA/cuda_header.h"

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_boundaries_thermo(float*input, Size size, float left, float right){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; z_iter++){ 
		float*cell = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
		if(y_iter == 0){
			if(x_iter < 3*(size.width_/4.f) && x_iter > (size.width_/4.f)){
				cell[0] = left;
			}else{
				cell[0] = right;
			}
		}
	}
}

extern "C"
void cuda_fluid_boundaries_thermo(void *input, Size size, float left, float right){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_boundaries_thermo<<<Dg,Db>>>((float *)input, size, left, right);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_fluid_project() failed to launch error = %d\n", error);
	}
}
#endif