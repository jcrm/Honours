#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_initial(unsigned char *input, float3 size_WHD, size_t pitch, size_t pitch_slice, float value){ 
	int x_iter = blockIdx.x*blockDim.x + threadIdx.x;
	int y_iter = blockIdx.y*blockDim.y + threadIdx.y;
	int z_iter = 0;

	for(z_iter = 0; z_iter < size_WHD.z; ++z_iter){ 
		//location is z slide + y position + variable size time x position
		unsigned char *cell_value = input + (z_iter*pitch_slice) + (y_iter*pitch) + (4*x_iter);
		cell_value[0] = signed int(value);
		cell_value[1] = signed int(0);
		cell_value[2] = signed int(0);
		cell_value[3] = signed int(0);
		if(x_iter +1 < size_WHD.x && x_iter - 1 > 0){
			if(y_iter + 1 < size_WHD.y && y_iter - 1 > 0){
				if(z_iter + 1 < size_WHD.z && z_iter - 1 > 0){
					cell_value[0] = signed int(0);
					cell_value[1] = signed int(0);
					cell_value[2] = signed int(0);
					cell_value[3] = signed int(0);
				}
			}
		}
	}
}
extern "C"
void cuda_fluid_initial(void *velocityinput, float3 size_WHD, size_t pitch, size_t pitch_slice, float value){
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

	cuda_kernel_initial<<<Dg,Db>>>((unsigned char *)velocityinput, size_WHD, pitch, pitch_slice, value);

	error = cudaGetLastError();
	if (error != cudaSuccess){
		printf("cuda_kernel_initial() failed to launch error = %d\n", error);
	}
}