#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <math.h>
#include "../Source/CUDA/cuda_header.h"



//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_bouyancy(unsigned char *output, float *input, float *input_two, Size size, Size size_two){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					unsigned char* output_velocity = output + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					float* input_thermo = input + (z_iter*size_two.pitch_slice_) + (y_iter*size_two.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
					float* input_water = input_two + (z_iter*size_two.pitch_slice_) + (y_iter*size_two.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
					float theta = input_thermo[theta_identifier_];
					float qv = input_water[qv_identifier_];
					float qh = input_water[qc_identifier_];
					//buoyancy
					output_velocity[y_identifier_] += 9.8f * (((theta*(1.f+(0.61f*qv))) / T0) - qh )* time_step;
				}
			}
		}
	}
}

extern "C"
void cuda_fluid_bouyancy(void *output, void *input, void *input_two, Size size, Size size_two){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_bouyancy<<<Dg,Db>>>((unsigned char *)output, (float *)input, (float *)input_two, size, size_two);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}