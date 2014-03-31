#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

__global__ void cuda_kernel_advect(unsigned char *output, unsigned char *velocityInput, float3 sizeWHD, size_t pitch, size_t pitchSlice){ 
	int xIter = blockIdx.x*blockDim.x + threadIdx.x;
	int yIter = blockIdx.y*blockDim.y + threadIdx.y;
	int zIter = 0;
	float timeStep = 1.f;

	 // in the case where, due to quantization into grids, we have
    // more threads than pixels, skip the threads which don't
    // correspond to valid pixels
    if (xIter >= sizeWHD.x || yIter >= sizeWHD.y) return;

	for(zIter = 0; zIter < sizeWHD.z; ++zIter){ 
		//location is z slide + y position + variable size time x position
		int location =(zIter*pitchSlice) + (yIter*pitch) + (4*xIter);
		unsigned char* cellVelocity = velocityInput + location;
		float3 pos;

		pos.x = xIter;
		pos.y = yIter;
		pos.z = zIter;

		pos.x -= (timeStep * cellVelocity[0]);
		pos.y -= (timeStep * cellVelocity[1]);
		pos.z -= (timeStep * cellVelocity[2]);
		
		pos.x /= sizeWHD.x;
		pos.y /= sizeWHD.y;
		pos.z += 0.5;
		pos.z /= sizeWHD.z;

		unsigned char* outputPixel = output + location;
		outputPixel[0] = pos.x;
		outputPixel[1] = pos.y;
		outputPixel[2] = pos.z;
		outputPixel[3] = 250; 

		cellVelocity[0] = outputPixel[0];
		cellVelocity[1] = outputPixel[1];
		cellVelocity[2] = outputPixel[2];
		cellVelocity[3] = outputPixel[3];

	}
}
extern "C"
void cuda_fluid_advect(void *output, void *velocityinput, float3 sizeWHD, size_t pitch, size_t pitchSlice){
	cudaError_t error = cudaSuccess;

    dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
    dim3 Dg = dim3((sizeWHD.x+Db.x-1)/Db.x, (sizeWHD.y+Db.y-1)/Db.y);

    cuda_kernel_advect<<<Dg,Db>>>((unsigned char *)output, (unsigned char *)velocityinput, sizeWHD, pitch, pitchSlice);

    error = cudaGetLastError();
    if (error != cudaSuccess){
        printf("cuda_kernel_advect() failed to launch error = %d\n", error);
    }
}