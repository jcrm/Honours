#ifndef _DISPLAY_CUDA_
#define _DISPLAY_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math.h"

#include "../Source/CUDA/cuda_header.h"

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_display(float *output, unsigned char* input, Size size, Size size_two){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		float data;
		unsigned char *cell_velocity = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
		float *display_output = output + (z_iter*size_two.pitch_slice_) + (y_iter*size_two.pitch_) + x_iter;
		float x = cell_velocity[x_identifier_] * cell_velocity[x_identifier_];
		float y = cell_velocity[y_identifier_] * cell_velocity[y_identifier_];
		float z = cell_velocity[z_identifier_] * cell_velocity[z_identifier_];

		float density = x + y + z;
		density = sqrt(density);
		cell_velocity[3] = density;
		display_output[0] = 0.0f;
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					if(density > 0){
						display_output[0] = density/220;
					}
				}
			}
		}
	}
}

extern "C"
void cuda_fluid_display(void *output, void *input, Size size, Size size_two){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_display<<<Dg,Db>>>((float *)output, (unsigned char *)input, size, size_two);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_fluid_project() failed to launch error = %d\n", error);
	}
}
#endif