#ifndef ADVECT_VELOCITY_CUDA_
#define ADVECT_VELOCITY_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "../Source/CUDA/cuda_header.h"

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_advect_velocity(float *output, float*input, Size size){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					float *fieldRight = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
					float *fieldDown = input + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					float*fieldRightCorner = input + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
					float *field = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);

					float *fieldRightBack = input + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
					float *fieldDownBack = input + ((z_iter+1)*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					float *fieldRightCornerBack = input + ((z_iter+1)*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
					float *fieldBack = input + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);

					float temp_X_1 = field[x_identifier_] +((fieldRight[x_identifier_]-field[x_identifier_])*0.5f);
					float temp_Y_1 = field[y_identifier_] +((fieldRight[y_identifier_]-field[y_identifier_])*0.5f);
					float temp_Z_1 = field[z_identifier_] +((fieldRight[z_identifier_]-field[z_identifier_])*0.5f);

					float temp_X_2 = fieldDown[x_identifier_] +((fieldRightCorner[x_identifier_]-fieldDown[x_identifier_])*0.5f);
					float temp_Y_2 = fieldDown[y_identifier_] +((fieldRightCorner[y_identifier_]-fieldDown[y_identifier_])*0.5f);
					float temp_Z_2 = fieldDown[z_identifier_] +((fieldRightCorner[z_identifier_]-fieldDown[z_identifier_])*0.5f);

					float temp_X_3 = fieldBack[x_identifier_] +((fieldRightBack[x_identifier_]-fieldBack[x_identifier_])*0.5f);
					float temp_Y_3 = fieldBack[y_identifier_] +((fieldRightBack[y_identifier_]-fieldBack[y_identifier_])*0.5f);
					float temp_Z_3 = fieldBack[z_identifier_] +((fieldRightBack[z_identifier_]-fieldBack[z_identifier_])*0.5f);

					float temp_X_4 = fieldDownBack[x_identifier_] +((fieldRightCornerBack[x_identifier_]-fieldDownBack[x_identifier_])*0.5f);
					float temp_Y_4 = fieldDownBack[y_identifier_] +((fieldRightCornerBack[y_identifier_]-fieldDownBack[y_identifier_])*0.5f);
					float temp_Z_4 = fieldDownBack[z_identifier_] +((fieldRightCornerBack[z_identifier_]-fieldDownBack[z_identifier_])*0.5f);

					temp_X_1 =(temp_X_1 + (temp_X_2-temp_X_1)*0.5f);
					temp_Y_1 =(temp_Y_1 + (temp_Y_2-temp_Y_1)*0.5f);
					temp_Z_1 =(temp_Z_1 + (temp_Z_2-temp_Z_1)*0.5f);

					temp_X_3 =(temp_X_3 + (temp_X_4-temp_X_3)*0.5f);
					temp_Y_3 =(temp_Y_3 + (temp_Y_4-temp_Y_3)*0.5f);
					temp_Z_3 =(temp_Z_3 + (temp_Z_4-temp_Z_3)*0.5f);

					float*output_velocity = output + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					output_velocity[x_identifier_] = temp_X_1 + ((temp_X_3-temp_X_1)*0.5f);
					output_velocity[y_identifier_] = temp_Y_1 + ((temp_Y_3-temp_Y_1)*0.5f);
					output_velocity[z_identifier_] = temp_Z_1 + ((temp_Z_3-temp_Z_1)*0.5f);
				}
			}
		}
		
	}
	
}

extern "C"
void cuda_fluid_advect_velocity(void *output, void *input, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_advect_velocity<<<Dg,Db>>>((float *)output, (float *)input, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}
#endif