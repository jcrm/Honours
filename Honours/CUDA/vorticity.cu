#ifndef _VORTICITY_CUDA_
#define _VORTICITY_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math.h"

#include "../Source/CUDA/cuda_header.h"

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_vorticity(float *output, float *input, Size size){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; z_iter++){ 
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					float*output_velocity = output + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					//vorticity confinement
					float scalar = 1.f;
					float*p = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					float*pLeft = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
					float*pRight = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
					float *pDown = input + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					float*pUp = input + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					float*pTop = input + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					float *pBottom = input + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);

					float3 curl_value = {
						((pUp[z_identifier_] - pDown[z_identifier_]) - (pBottom[y_identifier_] - pTop[y_identifier_])) / (2*dx), 
						((pBottom[x_identifier_] - pTop[x_identifier_]) - (pRight[z_identifier_] - pLeft[z_identifier_])) / (2*dx), 
						((pRight[y_identifier_] - pLeft[y_identifier_]) - (pUp[x_identifier_] - pDown[x_identifier_])) / (2*dx)
					};
					float temp = (curl_value.x * curl_value.x) + (curl_value.y * curl_value.y) + (curl_value.z * curl_value.z);
					if(temp == 0){
						temp = 1;
					}
					temp = sqrt(temp);
					float invLen = 1.0f / temp;
					float3 norm_value = {
						curl_value.x * invLen,
						curl_value.y * invLen,
						curl_value.z * invLen
					};
					float3 vect = {
						(norm_value.y * curl_value.z) - (norm_value.z * curl_value.y),
						(norm_value.z * curl_value.x) - (norm_value.x * curl_value.z),
						(norm_value.x * curl_value.y) - (norm_value.y * curl_value.x)
					};/*
					output_velocity[x_identifier_] += (vect.x * dx * scalar * time_step);
					output_velocity[y_identifier_] += (vect.y * dx * scalar * time_step);
					output_velocity[z_identifier_] += (vect.z * dx * scalar * time_step);*/
				}
			}
		}
	}
}

extern "C"
void cuda_fluid_vorticity(void *output, void *input, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_vorticity<<<Dg,Db>>>((float*)output, (float*)input, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}
#endif