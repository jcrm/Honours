#ifndef _DIVERGENCE_CUDA_
#define _DIVERGENCE_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../Source/CUDA/cuda_header.h"

//output diverrgnece texture //input velocity derrivitive teture
__global__ void cuda_kernel_divergence(float* output, float* input, Size size){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; z_iter++){ 
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					float*fieldRight = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
					float*fieldUp = input + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					float*fieldTop = input + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					float*fieldLeft = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
					float*fieldDown = input + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					float*fieldBottom = input + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);

					float*output_divergence = output + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					float temp_x = fieldLeft[x_identifier_];
					float temp_y = fieldDown[y_identifier_];
					float temp_z = fieldBottom[z_identifier_];

					float temp_x_2 = fieldRight[x_identifier_];
					float temp_y_2 = fieldUp[y_identifier_];
					float temp_z_2 = fieldTop[z_identifier_];

					float frl_x = temp_x_2 - temp_x;
					float frl_y = temp_y_2 - temp_y;
					float frl_z = temp_z_2 - temp_z;
					float value = (frl_x+frl_y+frl_z);
					value *= 0.5f;
					output_divergence[divergence_identifier_] = value;
					// Compute the velocity's divergence using central differences.  
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_divergence(void *divergence, void *input, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_divergence<<<Dg,Db>>>((float*)divergence, (float*)input, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_divergence() failed to launch error = %d\n", error);
	}
}
#endif