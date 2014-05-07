#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <math.h>
#include "../Source/CUDA/cuda_header.h"

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_vorticity(unsigned char *output, unsigned char *input, Size size){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size.depth_; ++z_iter){ 
		if(x_iter +1 < size.width_ && x_iter - 1 >= 0){
			if(y_iter + 1 < size.height_ && y_iter - 1 >= 0){
				if(z_iter + 1 < size.depth_ && z_iter - 1 >= 0){
					unsigned char*output_velocity = output + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					//vorticity confinement
					float scalar = 1.f;
					unsigned char *pLeft = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter-1));
					unsigned char *pRight = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * (x_iter+1));
					unsigned char *pDown = input + (z_iter*size.pitch_slice_) + ((y_iter-1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					unsigned char *pUp = input + (z_iter*size.pitch_slice_) + ((y_iter+1)*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter); 
					unsigned char *pTop = input + ((z_iter-1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
					unsigned char *pBottom = input + ((z_iter+1)*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);

					float3 curl_value = {
						((pDown[z_identifier_] - pUp[z_identifier_]) - (pBottom[y_identifier_] - pTop[y_identifier_])) / dx, 
						((pBottom[x_identifier_] - pTop[x_identifier_]) - (pRight[z_identifier_] - pLeft[z_identifier_])) / dx, 
						((pRight[y_identifier_] - pLeft[y_identifier_]) - (pDown[x_identifier_] - pUp[x_identifier_])) / dx
					};

					float invLen = 1.0f / sqrt((curl_value.x * curl_value.x) + (curl_value.y * curl_value.y) + (curl_value.z * curl_value.z));
					float3 norm_value = {
						curl_value.x * invLen,
						curl_value.y * invLen,
						curl_value.z * invLen
					};

					output_velocity[x_identifier_] += ((norm_value.y * curl_value.z) - (norm_value.z * curl_value.y)) * dx * scalar * time_step;
					output_velocity[y_identifier_] += ((norm_value.z * curl_value.x) - (norm_value.x * curl_value.z)) * dx * scalar * time_step;
					output_velocity[z_identifier_] += ((norm_value.x * curl_value.y) - (norm_value.y * curl_value.x)) * dx * scalar * time_step;
				}
			}
		}
		unsigned char *output_velocity = input + (z_iter*size.pitch_slice_) + (y_iter*size.pitch_) + (PIXEL_FMT_SIZE_RGBA * x_iter);
		if(x_iter +1 == size.width_ || x_iter - 1 < 0){
			if(y_iter == 0 && z_iter == 0){
				signed int x = output_velocity[x_identifier_] - 1;
				signed int y = output_velocity[y_identifier_] - 1;
				signed int z = output_velocity[z_identifier_] - 1;
				x =  x < 0 ? 0 : x;
				y =  y < 0 ? 0 : y;
				z =  z < 0 ? 0 : z;
				output_velocity[x_identifier_] = x;
				output_velocity[y_identifier_] = y;
				output_velocity[z_identifier_] = z;
			}
		}
		if(y_iter + 1 == size.height_ || y_iter - 1 < 0){
			if(x_iter == 0 && z_iter == 0){
				signed int x = output_velocity[x_identifier_] - 1;
				signed int y = output_velocity[y_identifier_] - 1;
				signed int z = output_velocity[z_identifier_] - 1;
				x =  x < 0 ? 0 : x;
				y =  y < 0 ? 0 : y;
				z =  z < 0 ? 0 : z;
				output_velocity[x_identifier_] = x;
				output_velocity[y_identifier_] = y;
				output_velocity[z_identifier_] = z;
			}
		}
		if(z_iter + 1 == size.depth_ || z_iter - 1 < 0){
			if(y_iter == 0 && x_iter == 0){
				signed int x = output_velocity[x_identifier_] - 1;
				signed int y = output_velocity[y_identifier_] - 1;
				signed int z = output_velocity[z_identifier_] - 1;
				x =  x < 0 ? 0 : x;
				y =  y < 0 ? 0 : y;
				z =  z < 0 ? 0 : z;
				output_velocity[x_identifier_] = x;
				output_velocity[y_identifier_] = y;
				output_velocity[z_identifier_] = z;
			}
		}
	}
}

extern "C"
void cuda_fluid_vorticity(void *output, void *input, Size size){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_vorticity<<<Dg,Db>>>((unsigned char *)output, (unsigned char *)input, size);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}