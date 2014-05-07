#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "../Source/CUDA/cuda_header.h"

//output diverrgnece texture //input velocity derrivitive teture
__global__ void cuda_kernel_divergence(unsigned char* output, unsigned char* input, Size size, int divergence_index){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					unsigned char *fieldLeft = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * (x_iter-1));
					unsigned char *fieldRight = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * (x_iter+1));
					unsigned char *fieldDown = input + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE * x_iter); 
					unsigned char *fieldUp = input + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE * x_iter); 
					unsigned char *fieldTop = input + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char *fieldBottom = input + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char *output_divergence = output + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE * x_iter);
					output_divergence[divergence_index] = signed int(0.5f * ((signed int(fieldRight[x_identifier_]) - signed int(fieldLeft[x_identifier_])) + 
						(signed int(fieldTop[y_identifier_]) - signed int(fieldBottom[y_identifier_])) + 
						(signed int(fieldUp[z_identifier_]) - signed int(fieldDown[z_identifier_]))));
					// Compute the velocity's divergence using central differences.  
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_divergence(void *divergence, void *input, Size size, int divergence_index){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_divergence<<<Dg,Db>>>((unsigned char *)divergence, (unsigned char *)input, size, divergence_index);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_divergence() failed to launch error = %d\n", error);
	}
}
