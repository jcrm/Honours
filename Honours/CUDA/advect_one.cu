#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#define timeStep 1.f
//output velocity derrivitive teture //input velcoity texutre
__global__ void cuda_kernel_advect_one_texture(unsigned char *input, float3 size_WHD, size_t pitch, size_t pitch_slice, float4 advect_index){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size_WHD.z; ++z_iter){ 
		if(x_iter +1 < size_WHD.x && x_iter - 1 > 0){
			if(y_iter + 1 < size_WHD.y && y_iter - 1 > 0){
				if(z_iter + 1 < size_WHD.z && z_iter - 1 > 0){
					//location is z slide + y position + variable size time x position
					unsigned char *current_velocity = input + (z_iter*pitch_slice) + (y_iter*pitch) + (4*x_iter);
					unsigned char *output_velocity = input + (z_iter*pitch_slice) + (y_iter*pitch) + (4*x_iter);

					float pos_x = float((x_iter +0.5f)- (timeStep * signed int(current_velocity[0])))/ size_WHD.x;
					float pos_y = float((y_iter +0.5f) - (timeStep * signed int(current_velocity[1])))/ size_WHD.y;
					float pos_z = float((z_iter +0.5f) - (timeStep * signed int(current_velocity[2]))+0.5f)/ size_WHD.z;

					unsigned int location = (pos_z*pitch_slice) + (pos_y*pitch) + (4*pos_x);
					current_velocity = input + location;

					output_velocity[0] = signed int(current_velocity[0]);
					output_velocity[1] = signed int(current_velocity[1]);
					output_velocity[2] = signed int(current_velocity[2]);
				}
			}
		}
		
	}
}

extern "C"
void cuda_fluid_advect_one_texture(void *output, void *input, float3 size_WHD, size_t pitch, size_t pitch_slice, float4 advect_index){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

	cuda_kernel_advect_one_texture<<<Dg,Db>>>((unsigned char *)input, size_WHD, pitch, pitch_slice, advect_index);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_advect() failed to launch error = %d\n", error);
	}
}