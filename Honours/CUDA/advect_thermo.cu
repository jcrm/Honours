#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "../Source/CUDA/cuda_header.h"

__global__ void cuda_kernel_advect_thermo(float *input, Size size){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					float *fieldRight = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * (x_iter+1));
					float *fieldDown = input + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter); 
					float *fieldRightCorner = input + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RG * (x_iter+1));
					float *field = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);

					float *fieldRightBack = input + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * (x_iter+1));
					float *fieldDownBack = input + ((z_iter+1)*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter); 
					float *fieldRightCornerBack = input + ((z_iter+1)*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RG * (x_iter+1));
					float *fieldBack = input + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);

					float temp_X_1 = field[theta_identifier_] +((fieldRight[theta_identifier_]-field[theta_identifier_])*0.5f);

					float temp_X_2 = fieldDown[theta_identifier_] +((fieldRightCorner[theta_identifier_]-fieldDown[theta_identifier_])*0.5f);

					float temp_X_3 = fieldBack[theta_identifier_] +((fieldRightBack[theta_identifier_]-fieldBack[theta_identifier_])*0.5f);

					float temp_X_4 = fieldDownBack[theta_identifier_] +((fieldRightCornerBack[theta_identifier_]-fieldDownBack[theta_identifier_])*0.5f);

					temp_X_1 =(temp_X_1 + (temp_X_2-temp_X_1)*0.5f);
					temp_X_3 =(temp_X_3 + (temp_X_4-temp_X_3)*0.5f);

					float *output_thermo = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RG * x_iter);
					output_thermo[theta_advect_identifier_] = temp_X_1 + ((temp_X_3-temp_X_1)*0.5f);
				}
			}
		}
	}
}

extern "C"
void cuda_fluid_advect_thermo(void *input, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_advect_thermo<<<Dg,Db>>>((float*)input, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}