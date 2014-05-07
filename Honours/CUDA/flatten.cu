#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "math.h"
#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_rain(unsigned char *output, float *input, Size size, Size size_two){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	float rain_sum = 0.f;
	for(z_iter = 0; z_iter < size.depth_; ++z_iter){
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					float* cellRain = input + (z_iter*size_two.pitch_slice_) + (y_iter*size_two.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
					rain_sum += cellRain[F_identifier_];
				}
			}
		}
	}
	unsigned char* rain = output + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
	if(rain_sum != 0){
		rain[0] = rain_sum;
	}
	rain[0] = rain_sum;
	rain[1] = 10;
	rain[2] = -5;
	rain[3] = 127;
}
extern "C"
void cuda_fluid_rain(void *output, void *input, Size size, Size size_two){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	//dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_rain<<<Dg,Db>>>((unsigned char *)output, (float *)input, size, size_two);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}