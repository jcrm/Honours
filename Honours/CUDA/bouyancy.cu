#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <math.h>
#include "../Source/CUDA/cuda_header.h"

#define vptemp 295.f
#define dx 1.f
#define time_step 1.f
#define PIXEL_FMT_SIZE 4
#define x_identifier_ 0
#define y_identifier_ 1
#define z_identifier_ 2

#define qv_identifier_ 0
#define qc_identifier_ 1
#define theta_identifier_ 2

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_bouyancy(unsigned char *output, unsigned char *input, unsigned char *input_two, Size size){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					unsigned char* output_velocity = output + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char* input_thermo = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char* inout_water = input_two + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
					float theta = input_thermo[theta_identifier_];
					float qv = inout_water[qv_identifier_];
					float qh = inout_water[qc_identifier_];
					//buoyancy
					output_velocity[y_identifier_] += 9.8f * (((theta*(1.f+(0.61f*qv))) / vptemp) - qh )* time_step;
				}
			}
		}
	}
}

extern "C"
void cuda_fluid_bouyancy(void *output, unsigned char *input, unsigned char *input_two, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_bouyancy<<<Dg,Db>>>((unsigned char *)output, (unsigned char *)input, (unsigned char *)input_two, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}