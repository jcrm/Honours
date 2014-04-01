#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_initial(unsigned char *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice, float value){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	for(zIter = 0; zIter < size_WHD.z; ++zIter){ 
		if(xIter > ((size_WHD.x/2) - 10) && xIter < ((size_WHD.x/2) + 10)){
			if(yIter > ((size_WHD.y/2) - 10) && yIter < ((size_WHD.y/2) + 10)){
				if(zIter > ((size_WHD.z/2) - 10) && zIter < ((size_WHD.z/2) + 10)){
					//location is z slide + y position + variable size time x position
					unsigned char* cellVelocity = velocityInput + (zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
					cellVelocity[0] = value;
					cellVelocity[1] = value;
					cellVelocity[2] = value;
					cellVelocity[3] = 1.0f;
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