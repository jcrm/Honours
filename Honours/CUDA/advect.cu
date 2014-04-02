#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_advect(unsigned char *output, unsigned char *velocityInput, float3 size_WHD, size_t pitch, size_t pitch_slice){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;
	float timeStep = 1.f;

	for(zIter = 0; zIter < size_WHD.z; ++zIter){ 
		//location is z slide + y position + variable size time x position
		int location =(zIter*pitch_slice) + (yIter*pitch) + (4*xIter);
		unsigned char* cellVelocity = velocityInput + location;
		float3 pos;
		pos.x = (xIter - (timeStep * cellVelocity[0]))/ size_WHD.x;
		pos.y = (yIter - (timeStep * cellVelocity[1]))/ size_WHD.y;
		pos.z = (zIter - (timeStep * cellVelocity[2])+0.5f)/ size_WHD.z;

		unsigned char* outputPixel = output + location;
		location =(pos.z*pitch_slice) + (pos.y*pitch) + (4*pos.x);
		cellVelocity = velocityInput + location;

		outputPixel[0] = cellVelocity[0];
		outputPixel[1] = cellVelocity[1];
		outputPixel[2] = cellVelocity[2];
		outputPixel[3] = 0; 
	}
}

extern "C"
void cuda_fluid_advect(void *output, void *velocityinput, float3 size_WHD, size_t pitch, size_t pitch_slice){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((size_WHD.x+Db.x-1)/Db.x, (size_WHD.y+Db.y-1)/Db.y);

    cuda_kernel_advect<<<Dg,Db>>>((unsigned char *)output, (unsigned char *)velocityinput, size_WHD, pitch, pitch_slice);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_advect() failed to launch error = %d\n", error);
    }
}