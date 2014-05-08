#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "math.h"
#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_rain(float *output, float *input, Size size, Size size_two){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;
	float rain_sum = 0.f;
	int ident = 0;
	for(z_iter = 0; z_iter < size_two.depth_; ++z_iter){
		if(x_iter +1 < size_two.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size_two.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size_two.depth_ && z_iter - 1 >= 0){
					float* cell_rain = input + (z_iter*size_two.pitch_slice_) + (y_iter*size_two.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
					rain_sum += cell_rain[F_identifier_];
				}
			}
		}
	}
	int xIter = x_iter;
	int yIter = y_iter;
	if(x_iter%2 != 0 && y_iter%2 != 0){
		xIter--;
		yIter--;
		ident = 3;
	}else if(x_iter%2 != 0){
		xIter--;
		ident = 1;
	}else if(y_iter%2 != 0){
		yIter--;
		ident = 2;
	}
	xIter /= 2;
	yIter /= 2;
	float* rain = output + (yIter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * xIter);
	rain[ident] = rain_sum;
}
extern "C"
void cuda_fluid_rain(void *output, void *input, Size size, Size size_two){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	//dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);
	dim3 Dg = dim3((size_two.width_+Db.x-1)/Db.x, (size_two.height_+Db.y-1)/Db.y);

	cuda_kernel_rain<<<Dg,Db>>>((float *)output, (float *)input, size, size_two);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_jacobi() failed to launch error = %d\n", error);
	}
}