#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#define PIXEL_FMT_SIZE 4
#define x_identifier_ 0
#define y_identifier_ 1
#define z_identifier_ 2
//output diverrgnece texture //input velocity derrivitive teture
__global__ void cuda_kernel_divergence(unsigned char* output, unsigned char* input,float3 size_WHD, size_t pitch, size_t pitch_slice, int divergence_index){
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size_WHD.z; ++z_iter){ 
		if(x_iter +1 < size_WHD.x && x_iter - 1 >= 0){
			if(y_iter + 1 < size_WHD.y && y_iter - 1 >= 0){
				if(z_iter + 1 < size_WHD.z && z_iter - 1 >= 0){
					unsigned char *fieldLeft = input + (z_iter*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * (x_iter-1));
					unsigned char *fieldRight = input + (z_iter*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * (x_iter+1));
					unsigned char *fieldDown = input + (z_iter*pitch_slice) + ((y_iter-1)*pitch) + (PIXEL_FMT_SIZE * x_iter); 
					unsigned char *fieldUp = input + (z_iter*pitch_slice) + ((y_iter+1)*pitch) + (PIXEL_FMT_SIZE * x_iter); 
					unsigned char *fieldTop = input + ((z_iter-1)*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char *fieldBottom = input + ((z_iter+1)*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * x_iter);
					unsigned char *output_divergence = output + (z_iter*pitch_slice) + (y_iter*pitch) + (PIXEL_FMT_SIZE * x_iter);
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
void cuda_fluid_divergence(void *divergence, void *input, float3 size_WHD, size_t pitch, size_t pitch_slice, int divergence_index){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

	cuda_kernel_divergence<<<Dg,Db>>>((unsigned char *)divergence, (unsigned char *)input,size_WHD, pitch, pitch_slice, divergence_index);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_divergence() failed to launch error = %d\n", error);
	}
}
