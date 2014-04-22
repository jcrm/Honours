#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <math.h> 
#define dx 1.f
#define time_step 1.f
//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_forces(unsigned char *output, unsigned char *input, float3 size_WHD, size_t pitch, size_t pitch_slice){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size_WHD.z; ++z_iter){ 
		if(x_iter +1 < size_WHD.x && x_iter - 1 > 0){
			if(y_iter + 1 < size_WHD.y && y_iter - 1 > 0){
				if(z_iter + 1 < size_WHD.z && z_iter - 1 > 0){
					unsigned char*output_velocity = output + (z_iter*pitch_slice) + (y_iter*pitch) + (4*x_iter);
					//vorticity confinement
					float scalar = 1.f;
					unsigned char *pLeft = input + (z_iter*pitch_slice) + (y_iter*pitch) + (4*(x_iter-1));
					unsigned char *pRight = input + (z_iter*pitch_slice) + (y_iter*pitch) + (4*(x_iter+1));
					unsigned char *pDown = input + (z_iter*pitch_slice) + ((y_iter-1)*pitch) + (4*x_iter); 
					unsigned char *pUp = input + (z_iter*pitch_slice) + ((y_iter+1)*pitch) + (4*x_iter); 
					unsigned char *pTop = input + ((z_iter-1)*pitch_slice) + (y_iter*pitch) + (4*x_iter);
					unsigned char *pBottom = input + ((z_iter+1)*pitch_slice) + (y_iter*pitch) + (4*x_iter);

					float3 curl_value = {
						((pDown[2] - pUp[2]) - (pBottom[1] - pTop[1])) / dx, 
						((pBottom[0] - pTop[0]) - (pRight[2] - pLeft[2])) / dx, 
						((pRight[1] - pLeft[1]) - (pDown[0] - pUp[0])) / dx
					};

					float invLen = 1.0f / sqrt((curl_value.x * curl_value.x) + (curl_value.y * curl_value.y) + (curl_value.z * curl_value.z));
					float3 norm_value = {
						curl_value.x * invLen,
						curl_value.y * invLen,
						curl_value.z * invLen
					};

					output_velocity[0] += ((norm_value.y * curl_value.z) - (norm_value.z * curl_value.y)) * dx * scalar * time_step;
					output_velocity[1] += ((norm_value.z * curl_value.x) - (norm_value.x * curl_value.z)) * dx * scalar * time_step;
					output_velocity[2] += ((norm_value.x * curl_value.y) - (norm_value.y * curl_value.x)) * dx * scalar * time_step;

					//buoyancy

				}
			}
		}
	}
}

extern "C"
void cuda_fluid_forces(void *output, void *input, float3 size_WHD, size_t pitch, size_t pitch_slice){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

	cuda_kernel_forces<<<Dg,Db>>>((unsigned char *)output, (unsigned char *)input, size_WHD, pitch, pitch_slice);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}