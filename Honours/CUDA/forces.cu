#ifndef _FORCES_CUDA_
#define _FORCES_CUDA_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "math.h"

#include "../Source/CUDA/cuda_header.h"

//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_force(float *output, float *input, Size size, float delta_time){ 
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
						p[x_identifier_],
						p[y_identifier_],
						p[z_identifier_]
					};

					float3 N_value = {
						pRight[3]-pLeft[3], 
						pUp[3] - pDown[3], 
						pBottom[3] - pTop[3]
					};

					float invLen = (N_value.x * N_value.x) + (N_value.y * N_value.y) + (N_value.z * N_value.z);
					invLen =  invLen == 0 ?  1 : invLen;
					invLen = sqrtf(invLen);
					N_value.x /= invLen;
					N_value.y /= invLen;
					N_value.z /= invLen;

					float3 vorticity = {
						(N_value.y *curl_value.z ) - (N_value.z * curl_value.y),
						(N_value.z * curl_value.x) - (N_value.x * curl_value.z),
						(N_value.x * curl_value.y) - (N_value.y * curl_value.x)
					};
					output_velocity[x_identifier_] += (vorticity.x * dx * scalar * time_step * delta_time);
					output_velocity[y_identifier_] += (vorticity.y * dx * scalar * time_step * delta_time);
					output_velocity[z_identifier_] += (vorticity.z * dx * scalar * time_step * delta_time);
				}
			}
		}
	}
}

extern "C"
void cuda_fluid_force(void *output, void *input, Size size, float delta_time){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size.width_+Db.x-1)/Db.x, (size.height_+Db.y-1)/Db.y);

	cuda_kernel_force<<<Dg,Db>>>((float*)output, (float*)input, size, delta_time);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}
#endif