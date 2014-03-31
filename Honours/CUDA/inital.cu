#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_initial(unsigned char *velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice, float value){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;

	// in the case where, due to quantization into grids, we have
	// more threads than pixels, skip the threads which don't
	// correspond to valid pixels
	if (xIter >= sizeWHD.x || yIter >= sizeWHD.y) return;

	for(zIter = 0; zIter < sizeWHD.z; ++zIter){ 
		if(xIter > ((sizeWHD.x/2) - 10) && xIter < ((sizeWHD.x/2) + 10)){
			if(yIter > ((sizeWHD.y/2) - 10) && yIter < ((sizeWHD.y/2) + 10)){
				if(zIter > ((sizeWHD.z/2) - 10) && zIter < ((sizeWHD.z/2) + 10)){
					//location is z slide + y position + variable size time x position
					unsigned char* cellVelocity = velocityInput + (zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
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
void cuda_fluid_initial(void *velocityinput, float3 sizeWHD, size_t pitch, size_t pitchSlice, float value){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((sizeWHD.x+Db.x-1)/Db.x, (sizeWHD.y+Db.y-1)/Db.y);

    cuda_kernel_initial<<<Dg,Db>>>((unsigned char *)velocityinput, sizeWHD, pitch, pitchSlice, value);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_initial() failed to launch error = %d\n", error);
    }
}